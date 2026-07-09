import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]

import dgl.function as fn  # type: ignore[import]


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion of distances.

    Expands a scalar distance into a vector of Gaussian basis values,
    giving the network a richer representation of spatial relationships
    than a single scalar (as in SchNet, DimeNet).

    Parameters
    ----------
    num_rbf : int
        Number of Gaussian basis functions.
    cutoff : float
        Maximum distance for the RBF centers.
    learnable : bool
        If True, centers and widths are learnable parameters.
    """

    def __init__(self, num_rbf=16, cutoff=10.0, learnable=False):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        widths = torch.full((num_rbf,), (cutoff / num_rbf))
        if learnable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, dist):
        """
        Parameters
        ----------
        dist : Tensor of shape (..., 1)

        Returns
        -------
        Tensor of shape (..., num_rbf)
        """
        return torch.exp(-((dist - self.centers) ** 2) / (self.widths ** 2))


class EGNNLayer(nn.Module):
    """E(n) equivariant graph convolution layer with attention and
    optional coordinate updates.

    Improvements over vanilla EGNN (Satorras et al. 2021):
    - RBF distance encoding instead of raw scalar distance
    - Pre-LayerNorm residual connections
    - Attention-weighted message aggregation
    - Optional coordinate updates with velocity clamping

    Parameters
    ----------
    hidden_size : int
        Hidden / output feature dimension (kept equal for clean residuals).
    edge_feat_size : int
        Dimension of input edge features (0 = none).
    dropout : float
        Dropout probability.
    num_rbf : int
        Number of Gaussian RBF centers for distance expansion.
    rbf_cutoff : float
        Maximum distance for RBF centers.
    update_coords : bool
        Whether to update node coordinates.
    use_attention : bool
        Whether to use attention-weighted aggregation.
    """

    def __init__(
        self,
        hidden_size,
        edge_feat_size=0,
        dropout=0.1,
        num_rbf=16,
        rbf_cutoff=10.0,
        update_coords=False,
        use_attention=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_feat_size = edge_feat_size
        self.update_coords = update_coords
        self.use_attention = use_attention

        self.rbf = GaussianRBF(num_rbf, rbf_cutoff)

        # Pre-norm
        self.norm = nn.LayerNorm(hidden_size)

        # Edge MLP: phi_e
        # TODO: add LayerNorm after first SiLU once existing EGNN checkpoints
        # are retrained (adds new state_dict keys, breaks strict loading)
        edge_in = hidden_size * 2 + num_rbf + edge_feat_size
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_size),
            nn.SiLU(),
            # nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

        # Node MLP: phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.SiLU(),
            # nn.LayerNorm(hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Attention
        if use_attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        # Coordinate update
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1, bias=False),
            )
            # Initialize near zero for stable start
            last_layer = self.coord_mlp[-1]
            if hasattr(last_layer, 'weight'):
                nn.init.zeros_(last_layer.weight)

    def message(self, edges):
        """Compute edge messages."""
        # RBF-encoded distance
        rbf = self.rbf(edges.data["dist"])  # (M, num_rbf)

        parts = [edges.src["h"], edges.dst["h"], rbf]
        if self.edge_feat_size > 0:
            edge_attr = edges.data["a"]
            if edge_attr.dim() > 2:
                edge_attr = edge_attr.squeeze(-1)
            parts.append(edge_attr)

        f = torch.cat(parts, dim=-1)
        msg_h = self.edge_mlp(f)

        result = {"msg_h": msg_h}

        if self.use_attention:
            att = self.att_mlp(msg_h)  # (M, 1)
            result["att"] = att

        if self.update_coords:
            coord_weight = self.coord_mlp(msg_h)  # (M, 1)
            # Clamp to prevent coordinate explosion
            coord_weight = torch.clamp(coord_weight, min=-10.0, max=10.0)
            result["msg_x"] = coord_weight * edges.data["x_diff"]

        return result

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """
        Parameters
        ----------
        graph : DGLGraph
        node_feat : Tensor (N, hidden_size)
        coord_feat : Tensor (N, 3)
        edge_feat : Tensor (M, edge_feat_size), optional

        Returns
        -------
        node_feat_out : Tensor (N, hidden_size)
        coord_feat_out : Tensor (N, 3) -- only if update_coords=True
        """
        with graph.local_scope():
            # Pre-norm
            h = self.norm(node_feat)

            graph.ndata["h"] = h
            graph.ndata["x"] = coord_feat

            if self.edge_feat_size > 0:
                graph.edata["a"] = edge_feat

            # Pairwise distances
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["dist"] = graph.edata["x_diff"].norm(dim=-1, keepdim=True)

            # Normalize direction vectors for coordinate updates
            if self.update_coords:
                graph.edata["x_diff"] = graph.edata["x_diff"] / (
                    graph.edata["dist"] + 1e-7
                )

            # Messages
            graph.apply_edges(self.message)

            # Aggregate node messages
            if self.use_attention:
                graph.edata["msg_h"] = graph.edata["msg_h"] * graph.edata["att"]

            graph.update_all(
                fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh")
            )
            h_neigh = graph.ndata["h_neigh"]

            # Node update with residual
            h_new = self.node_mlp(torch.cat([h, h_neigh], dim=-1))
            node_feat_out = node_feat + h_new  # residual on pre-norm input

            # Coordinate update
            if self.update_coords:
                graph.update_all(
                    fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh")
                )
                coord_feat_out = coord_feat + graph.ndata["x_neigh"]
                return node_feat_out, coord_feat_out

            return node_feat_out
