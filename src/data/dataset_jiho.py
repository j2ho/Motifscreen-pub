# src/data/training_dataset.py

from builtins import print
import os
import copy
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pickle
import torch
import dgl
import scipy.spatial
import scipy.sparse

# Local imports
import src.data.types as types
import src.data.utils as myutils
import src.data.kappaidx as kappaidx
from configs.config_loader import Config, DataPathsConfig, GraphParamsConfig, DataProcessingConfig, DataAugmentationConfig, CrossValidationConfig 

logger = logging.getLogger(__name__)


class MolecularLoader:
    def __init__(self, config_paths: DataPathsConfig, config_processing: DataProcessingConfig, config_augmentation: DataAugmentationConfig):
        self.config_paths = config_paths
        self.config_processing = config_processing
        self.config_augmentation = config_augmentation # Augmentation can be passed down if needed in loader
        self.mol2_cache = {}

    def load_keyatoms(self, keyatomf: str, targetname: str) -> Dict[str, List[str]]:
        try:
            data = np.load(keyatomf, allow_pickle=True)
            if 'keyatms' in data:
                return data['keyatms'].item()
            elif targetname in data: 
                return data
            return data
        except Exception as e:
            logger.error(f"Failed to load keyatoms from {keyatomf}: {e}")
            return {}
                
    def find_keyatomf(self, pname: str, source: str) -> Optional[str]:
        # Access config via self.config_paths
        # search_paths = [f'{self.config_paths.datapath}/{source}/{self.config_paths.keyatomf}',
        #                 f'{self.config_paths.datapath}/{source}/{pname}/{pname}.{self.config_paths.keyatomf}',
        # ]
        batch_subdir = getattr(self.config_paths, 'chembl_batch_subdir', 'batch_mol2s')
        search_paths = [f'{self.config_paths.datapath}/{source}/{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/{pname}.{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/{batch_subdir}/{pname}.{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/batch_mol2s/{pname}.{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/batch_mol2s_sim_check/{pname}.{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/batch_mol2s_qh/{pname}.{self.config_paths.keyatomf}',
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path

        logger.error(f"Cannot find keyatomf in {search_paths}")
        return None

    def read_mol2_single(self, mol2_path: str) -> Tuple:
        # Access config via self.config_processing
        if not os.path.exists(mol2_path):
            logger.error(f"MOL2 file not found: {mol2_path}")
            return None
            
        # Check if file is empty
        if os.path.getsize(mol2_path) == 0:
            logger.warning(f"MOL2 file is empty: {mol2_path}")
            return None
            
        try:
            return myutils.read_mol2(mol2_path, drop_H=self.config_processing.drop_H)
        except Exception as e:
            logger.error(f"Failed to read MOL2 {mol2_path}: {e}")
            return None

    def read_mol2_batch(self, mol2_path: str, tags: List[str] = None) -> Dict:
        # Access config via self.config_processing
        if self.config_processing.store_memory and mol2_path in self.mol2_cache:
            return self._extract_from_cache(mol2_path, tags)
        try:
            data = myutils.read_mol2_batch(
                mol2_path,
                drop_H=self.config_processing.drop_H,
                tags_read=tags
            )
            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read =data
            if self.config_processing.store_memory:
                self._cache_mol2_data(mol2_path, data)
            return data
        except Exception as e:
            logger.error(f"Failed to read batch MOL2 {mol2_path}: {e}")
            return None

    def _cache_mol2_data(self, mol2_path: str, data: Tuple):
        elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = data
        cache_data = {}
        for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read
        ):
            cache_data[tag] = (elem, q, bond, border, coord, nneigh, atm, atype)
        self.mol2_cache[mol2_path] = cache_data
        logger.info(f"Cached MOL2: {mol2_path}, molecules: {len(cache_data)}")

    def _extract_from_cache(self, mol2_path: str, tags: List[str]) -> Tuple:
        cache_data = self.mol2_cache[mol2_path]
        elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = [], [], [], [], [], [], [], [], []
        for tag in tags:
            if tag in cache_data:
                elem, q, bond, border, coord, nneigh, atm, atype = cache_data[tag]
                elems.append(elem)
                qs.append(q)
                bonds.append(bond)
                borders.append(border)
                xyz.append(coord)
                nneighs.append(nneigh)
                atms.append(atm)
                atypes.append(atype)
                tags_read.append(tag)
        return elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read

class GraphBuilder:
    """Constructs DGL graphs from molecular data"""
    def __init__(self, config_graph: GraphParamsConfig, config_augmentation: DataAugmentationConfig, config_processing: DataProcessingConfig, static: bool):
        self.config_graph = config_graph
        self.config_augmentation = config_augmentation
        self.config_processing = config_processing
        self.static = static

    def build_ligand_graph(self, mol_data: Tuple, name: str = "") -> dgl.DGLGraph:
        try:
            elems, qs, bonds, borders, xyz, nneighs, atypes = mol_data
            u, v, dX = self._build_edges(xyz, bonds,
                                   mode=self.config_graph.edgemode,
                                   top_k=self.config_graph.edgek[0],
                                   dcut=self.config_graph.edgedist[0])
            graph = dgl.graph((u, v))
            node_features = self._compute_ligand_node_features(elems, qs, nneighs, xyz, atypes)
            graph.ndata['attr'] = torch.tensor(node_features).float()
            edge_features = self._compute_ligand_edge_features(bonds, borders, u, v, xyz)
            graph.edata['attr'] = edge_features
            graph.ndata['x'] = torch.tensor(xyz).float()[:, None, :]
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]
            global_features = self._compute_global_features(elems, qs, bonds, borders, xyz, atypes)
            setattr(graph, "gdata", torch.tensor(global_features).float())
            graph.ndata['Y'] = torch.zeros(len(elems), 3)

            if (graph.in_degrees() == 0).any():
                import traceback
                logger.error(f"Graph {name} contains 0-in-degree")
                #traceback.print_exc()
                return None
            
            return graph
        except Exception as e:
            import traceback
            logger.error(f"Failed to build ligand graph for {name}: {e}")
            traceback.print_exc()
            return None

    def build_receptor_graph(self, npz_path: str, grids: np.ndarray, origin: torch.Tensor,
                           gridchain: str = None) -> Tuple[dgl.DGLGraph, np.ndarray, np.ndarray]:
        try:
            prop = np.load(npz_path)
            xyz = prop['xyz_rec']
            charges_rec = prop['charge_rec']
            atypes_rec = prop['atypes_rec']
            anames = prop['atmnames']
            aas_rec = prop['aas_rec']
            sasa_rec = prop['sasa_rec']
            bnds = prop['bnds_rec']

            xyz, grids = self._process_coordinates(xyz, grids, origin, gridchain)

            if self.config_augmentation.randomize > 1e-3:
                xyz = self._randomize_coordinates(xyz, self.config_augmentation.randomize)

            node_features = self._compute_receptor_node_features(
                xyz, grids, atypes_rec, anames, aas_rec, sasa_rec, charges_rec
            )

            graph = self._build_receptor_graph_structure(
                xyz, grids, node_features, bnds, anames
            )

            if graph is None:
                return None, None, None

            if self.config_graph.firstshell_as_grid:
                grids, grid_indices = self._apply_first_shell_logic(graph, xyz, grids)
            else:
                grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]

            return graph, grids, grid_indices

        except Exception as e:
            logger.error(f"Failed to build receptor graph from {npz_path}: {e}")
            return None, None, None

    def _build_edges(self, xyz: np.ndarray, bonds: List[List[int]],
                    mode: str, top_k: int, dcut: float) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        # Debug: Check input shape
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            logger.error(f"Unexpected xyz shape: {xyz.shape}, expected (N, 3)")
            raise ValueError(f"xyz must be shape (N, 3), got {xyz.shape}")

        X = torch.tensor(xyz[None, :])  # Shape: [1, N, 3]
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)  # Shape: [1, N, N, 3]

        # Debug: Check dX shape before distance calculation
        if dX.ndim != 4:
            logger.error(f"dX has unexpected shape: {dX.shape}, expected 4D tensor")
            logger.error(f"Original X shape: {X.shape}, xyz shape: {xyz.shape}")
            raise ValueError(f"dX dimension error: got {dX.ndim}D, expected 4D")

        u, v, d = self._find_distance_neighbors(dX, mode, top_k, dcut)
        return u, v, dX

    def _find_distance_neighbors(self, dX: torch.Tensor, mode: str = 'dist',
                               top_k: int = 8, dcut: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        if mode == 'dist':
            _, u, v = torch.where(D < dcut)
        elif mode == 'distT':
            mask = torch.where(torch.tril(D) < 1e-6, 100.0, 1.0)
            _, u, v = torch.where(mask * D < dcut)
        elif mode == 'topk':
            top_k_var = min(D.shape[1], top_k + 1)
            D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
            E_idx = E_idx[:, :, 1:]
            u = torch.arange(E_idx.shape[1])[:, None].repeat(1, E_idx.shape[2]).reshape(-1)
            v = E_idx[0].reshape(-1)
            in_degrees = torch.zeros(D.shape[1])
            in_degrees.scatter_add_(0, v, torch.ones_like(v, dtype=torch.float))
            zero_degree_nodes = (torch.where(in_degrees == 0)[0])
            if len(zero_degree_nodes) > 0: 
                for node in zero_degree_nodes:
                    closest_node = torch.argmin(D[0, node]) 
                    u = torch.cat([u, torch.tensor([closest_node])]) 
                    v = torch.cat([v, torch.tensor([node])])
        return u, v, D[0]

    def _compute_ligand_node_features(self, elems: List, qs: List, nneighs: np.ndarray,
                                    xyz: np.ndarray, atypes: List) -> np.ndarray:
        features = []
        ns = np.sum(nneighs, axis=1)[:, None] + 0.01
        ns = np.repeat(ns, 4, axis=1)
        normalized_nneighs = nneighs * (1.0 / ns)
        features.append(normalized_nneighs)
        sasa, nsasa, occl = myutils.sasa_from_xyz(xyz, elems)
        features.append(nsasa[:, None])
        features.append(occl[:, None])
        features.append(np.array(qs)[:, None])
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]
        elem_onehot = np.eye(len(myutils.ELEMS))[elem_indices]
        features.append(elem_onehot)

        return np.concatenate(features, axis=-1)

    def _compute_ligand_edge_features(self, bonds: List, borders: List,
                                    u: np.ndarray, v: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        bond_matrix = np.zeros((len(xyz), len(xyz)), dtype=int)
        bond_indices = np.zeros((len(xyz), len(xyz)), dtype=int)
        for k, (i, j) in enumerate(bonds):
            bond_indices[i, j] = bond_indices[j, i] = k
            bond_matrix[i, j] = bond_matrix[j, i] = 1
        bond_orders = np.zeros(len(u), dtype=np.int64)
        for k, (i, j) in enumerate(zip(u, v)):
            if bond_matrix[i, j]:
                bond_orders[k] = borders[bond_indices[i, j]]
        edge_features = torch.eye(5)[bond_orders]
        bond_graph = scipy.sparse.csgraph.shortest_path(
            scipy.sparse.csr_matrix(bond_matrix), directed=False, method='D')
        topo_distances = torch.tensor(bond_graph)[u, v]
        edge_features[:, -1] = 1.0 / (topo_distances + 0.00001)
        return edge_features

    def _compute_global_features(self, elems: List, qs: List, bonds: List,
                               borders: List, xyz: np.ndarray, atypes: List) -> np.ndarray:
        nflextors = 0
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]
        for bond_idx, (i, j) in enumerate(bonds):
            if elem_indices[i] != 1 and elem_indices[j] != 1 and borders[bond_idx] > 1:
                nflextors += 1
        kappa = kappaidx.calc_Kappaidx(atypes, bonds, False)
        normalized_kappa = [(kappa[0] - 40.0) / 40.0, (kappa[1] - 15.0) / 15.0, (kappa[2] - 10.0) / 10.0]
        com = np.mean(xyz, axis=0)
        centered_xyz = xyz - com
        inertia = np.dot(centered_xyz.transpose(), centered_xyz)
        eigvals, _ = np.linalg.eig(inertia)
        principal_values = (np.sqrt(np.sort(eigvals)) - 20.0) / 20.0
        natm = (len([e for e in elem_indices if e > 1]) - 25.0) / 25.0
        naro = (len([at for at in atypes if at in ['C.ar', 'C.aro', 'N.ar']]) - 6.0) / 6.0
        nacc, ndon = -1.0, -1.0
        for i, elem_idx in enumerate(elem_indices):
            if elem_idx not in [3, 4]:
                continue
            neighbors = [j for i_bond, j_bond in bonds if (i_bond == i or j_bond == i)]
            has_hydrogen = any(elem_indices[n] == 1 for n in neighbors if n != i)
            if has_hydrogen:
                ndon += 0.2
            else:
                nacc += 0.2
        nflextors_onehot = np.eye(10)[min(9, nflextors)]
        global_features = np.concatenate([
            nflextors_onehot, normalized_kappa, [nacc, ndon, naro], list(principal_values)
        ])
        return global_features

    def _process_coordinates(self, xyz: np.ndarray, grids: np.ndarray,
                           origin: torch.Tensor, gridchain: str = None) -> Tuple[np.ndarray, np.ndarray]:
        if gridchain is not None:
            pass
        origin_np = origin.squeeze().numpy()
        xyz_centered = xyz - origin_np
        grids_centered = grids - origin_np 
        return xyz_centered, grids_centered

    def _randomize_coordinates(self, xyz: np.ndarray, randomize_factor: float) -> np.ndarray:
        rand_xyz = 2.0 * randomize_factor * (0.5 - np.random.rand(*xyz.shape))
        return xyz + rand_xyz

    def _compute_receptor_node_features(self, xyz: np.ndarray, grids: np.ndarray,
                                      atypes_rec: List, anames: List, aas_rec: List,
                                      sasa_rec: List, charges_rec: List) -> np.ndarray:
        ngrids = len(grids)
        all_aas = np.concatenate([aas_rec, [0] * ngrids])
        all_xyz = np.concatenate([xyz, grids])
        atypes = np.array([types.find_gentype2num(at) for at in atypes_rec])
        all_atypes = np.concatenate([atypes, [0] * ngrids])
        all_sasa = np.concatenate([sasa_rec, [0.0] * ngrids])
        all_charges = np.concatenate([charges_rec, [0.0] * ngrids])
        #drop this
        #d2o = np.sqrt(np.sum(all_xyz * all_xyz, axis=1))
        d2o = np.zeros(all_xyz.shape[0]) # dummy
        
        features = []
        aa_onehot = np.eye(types.N_AATYPE)[all_aas]
        features.append(aa_onehot)
        atype_onehot = np.eye(max(types.gentype2num.values()) + 1)[all_atypes]
        features.append(atype_onehot)
        features.extend([
            all_sasa[:, None],
            all_charges[:, None],
            d2o[:, None]
        ])
        return np.concatenate(features, axis=-1)

    def _build_receptor_graph_structure(self, xyz: np.ndarray, grids: np.ndarray,
                                      node_features: np.ndarray, bonds: List,
                                      anames: List) -> Optional[dgl.DGLGraph]:
        try:
            all_xyz = np.concatenate([xyz, grids])
            kd = scipy.spatial.cKDTree(all_xyz)
            kd_grids = scipy.spatial.cKDTree(grids)

            ball_radius =  self.config_graph.ball_radius
            if not self.static:
                ball_radius = ball_radius + self.config_graph.ball_radius_var*2.0*(np.random.random()-0.5)
            indices = np.concatenate(kd_grids.query_ball_tree(kd, ball_radius))
            selected_indices = list(np.unique(indices).astype(np.int16))
            if len(selected_indices) > self.config_graph.maxnode:
                logger.error(f"Receptor nodes {len(selected_indices)} exceeds max {self.config_graph.maxnode}")
                return None
            bond_matrix = np.zeros((len(all_xyz), len(all_xyz)))
            index_map = {idx: i for i, idx in enumerate(selected_indices)}
            for i, j in bonds:
                if i in index_map and j in index_map:
                    k, l = index_map[i], index_map[j]
                    bond_matrix[k, l] = bond_matrix[l, k] = 1
            # Uncomment if want to ensure self-loops
            # for i in range(len(all_xyz)):
            #     bond_matrix[i, i] = 1
            selected_xyz = all_xyz[selected_indices]
            selected_xyz = torch.tensor(selected_xyz).float()
            X = selected_xyz[None, :].clone().detach()
            dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
            u, v, D = self._find_distance_neighbors(
                dX,
                mode=self.config_graph.edgemode,
                top_k=self.config_graph.edgek[1],
                dcut=self.config_graph.edgedist[1]
            )
            natm = len(xyz)
            within_cutoff = (D < self.config_graph.edgedist[1])
            within_cutoff[:natm, :] = within_cutoff[:, :natm] = 1
            valid_edges = within_cutoff[u, v]
            u = u[valid_edges >= 0]
            v = v[valid_edges >= 0]
            graph = dgl.graph((u, v))
            graph.ndata['attr'] = torch.tensor(node_features[selected_indices]).float()
            graph.ndata['x'] = selected_xyz.clone().detach()[:, None, :]
            edge_distances = torch.sqrt(torch.sum((selected_xyz[v] - selected_xyz[u]) ** 2, dim=-1) + 1e-6)[:, None]
            edge_features = self._distance_feature(edge_distances, 0.5, 5.0)
            bond_info = torch.tensor([bond_matrix[v_i, u_i] for u_i, v_i in zip(u, v)]).float()
            edge_features[:, 0] = bond_info
            grid_neighbors = ((u >= natm) * (v >= natm)).float()
            edge_features[:, 1] = grid_neighbors
            graph.edata['attr'] = edge_features
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]
            if graph.number_of_edges() > self.config_graph.maxedge:
                logger.error(f"Receptor edges {graph.number_of_edges()} exceeds max {self.config_graph.maxedge}")
                return None
            return graph
        except Exception as e:
            logger.error(f"Error building receptor graph structure: {e}")
            return None

    def _apply_first_shell_logic(self, graph: dgl.DGLGraph, xyz: np.ndarray,
                               grids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config_graph.firstshell_as_grid:
            grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]
            return grids, grid_indices
        recidx = np.where(graph.ndata['attr'][:, 0] == 0)[0]
        grididx = np.where(graph.ndata['attr'][:, 0] == 1)[0]
        all_xyz = graph.ndata['x'].squeeze(1).numpy()
        kd_g = scipy.spatial.cKDTree(all_xyz[grididx])
        kd_r = scipy.spatial.cKDTree(all_xyz[recidx])
        firstshell_idx = np.unique(np.concatenate(kd_g.query_ball_tree(kd_r, 4.5)))
        extended_grididx = np.concatenate([grididx, firstshell_idx]).astype(np.int32)
        extended_grids = all_xyz[extended_grididx]
        return extended_grids, extended_grididx

    def _distance_feature(self, distances: torch.Tensor, binsize: float = 0.5,
                         maxd: float = 5.0) -> torch.Tensor:
        d0 = 0.5 * (maxd - binsize)
        m = 5.0 / d0
        feat = 1.0 / (1.0 + torch.exp(-m * (distances - d0)))
        return feat.repeat(1, 3)

class TrainingDataSet(torch.utils.data.Dataset):
    def __init__(self, targets: List[str], ligands: List = None,
                 static: bool = False,
                 config: Config = None): 
        """
        Initialize training dataset with grouped parameters

        Args:
            targets: List of target identifiers
            ligands: List of ligand specifications
            config: Main configuration (from configs.config)
        """
        if config is None:
            raise ValueError("Config object must be provided to TrainingDataSet.")
        self.config = config
        self.static = static #true if used as validation

        self.targets = targets
        self.ligands = ligands
        self.decoys = self._load_decoys()

        # Initialize components by passing relevant sub-configs
        self.loader = MolecularLoader(
            config_paths=self.config.paths,
            config_processing=self.config.processing,
            config_augmentation=self.config.augmentation # Passed to loader for randomize_grid
        )
        self.graph_builder = GraphBuilder(
            config_graph=self.config.graph,
            config_augmentation=self.config.augmentation,
            config_processing=self.config.processing, # Passed to graph_builder for drop_H in ligand graph
            static=self.static
        )

        self.crossactives = self._load_crossactives() if self.config.cross_validation.load_cross else {}
        self._hard_negatives = {}
        logger.info(f"Training dataset initialized with {len(targets)} targets")
        logger.info(f"Graph config: {self.config.graph}")
        logger.info(f"Processing config: {self.config.processing}")
        logger.info(f"Augmentation config: {self.config.augmentation}")

    def __len__(self) -> int:
        return len(self.targets)

    def set_hard_negatives(self, hard_neg_dict: Dict[str, List[str]], n_slots: int = 5):
        """Set curriculum hard negatives for next epoch.

        Args:
            hard_neg_dict: {target_key: [decoy_id, ...]} sorted by descending score.
                           target_key is "source.pname".
            n_slots: number of batch slots to reserve for hard negatives.
        """
        self._hard_negatives = hard_neg_dict
        self._hard_neg_slots = n_slots

    def __getitem__(self, index: int) -> Optional[Tuple]:
        try:
            return self._get_item_safe(index)
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            # traceback.print_exc() # Uncomment for more detailed debug
            return self._get_null_result(self.targets[index])

    def _get_item_safe(self, index: int) -> Optional[Tuple]:
        target = self.targets[index]
        source = target.split('/')[0]
        pname = target.split('/')[1]

        receptor_file_paths = self._setup_file_paths(pname, source)
        if not self._validate_files(receptor_file_paths):
            return self._get_null_result(pname)

        keyatomf = self.loader.find_keyatomf(pname, source)
        if not keyatomf:
            return self._get_null_result(pname)
        keyatoms_dict = self.loader.load_keyatoms(keyatomf, targetname=pname)
        #print (f"Loaded keyatoms for {pname}: {keyatoms_dict}")

        grid_data = self._load_grid_data(receptor_file_paths['gridinfo'])
        if grid_data is None:
            print (f"Grid data missing for {pname}")
            return self._get_null_result(pname)
        grids, cats, mask = grid_data

        ligand_result, is_near_native = self._process_ligands(target, self.ligands[index], keyatoms_dict)
        if ligand_result is None:
            return self._get_null_result(pname)
        ligand_graphs, native_graph, key_xyz, key_indices, binding_labels, ligand_info = ligand_result
        origin = torch.tensor(np.mean(grids, axis=0)).float()

        # Transform key_xyz from ligand-COM frame to grid-centered frame
        # key_xyz is currently: (PDB coords) - ligand_COM
        # We need: (PDB coords) - grid_origin = key_xyz + ligand_COM - grid_origin
        native_com = ligand_info.get('native_com')
        if native_com is not None and native_graph is not None:
            key_xyz = key_xyz + native_com.view(1, 3) - origin.view(1, 3)
        
        gridchain = None # Still using None for now
        receptor_graph, processed_grids, grid_indices = self.graph_builder.build_receptor_graph(
            receptor_file_paths['propnpz'], grids, origin, gridchain
        )

        if receptor_graph is None:
            return self._get_null_result(pname)

        if (receptor_graph.number_of_edges() > self.config.graph.maxedge or
            receptor_graph.number_of_nodes() > self.config.graph.maxnode):
            logger.error(f"Graph too large for {target}: {receptor_graph.number_of_edges()} edges, {receptor_graph.number_of_nodes()} nodes")
            return self._get_null_result(pname)

        info = self._build_info_dict(pname, target, origin, processed_grids,
                                   grid_indices, receptor_file_paths['gridinfo'], ligand_info, is_near_native)

        return (receptor_graph, ligand_graphs, cats, mask, key_xyz,
               key_indices, binding_labels, info)

    def _setup_file_paths(self, pname, source: str) -> Dict[str, str]:
        if source == 'chembl':
            propnpz = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.prop.npz"
            gridinfo = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.grid.npz"
        else: 
            propnpz = f"{self.config.paths.datapath}/{source}/{pname}/{pname}_protein.prop.npz"
            gridinfo = f"{self.config.paths.datapath}/{source}/{pname}/{pname}_protein.grid.npz"

        parentpath = '/'.join(gridinfo.split('/')[:-1]) + '/'
        return {
            'propnpz': propnpz,
            'gridinfo': gridinfo,
            'parentpath': parentpath
        }

    def _validate_files(self, file_paths: Dict[str, str]) -> bool:
        required_files = ['gridinfo', 'propnpz']
        for file_key in required_files:
            if not os.path.exists(file_paths[file_key]):
                logger.error(f"Required file missing: {file_paths[file_key]}")
                return False
        return True

    def _load_grid_data(self, gridinfo_path: str) -> Optional[Tuple]:
        grids = []
        cats = []
        try:
            #print (f"Loading grid data from {gridinfo_path}")
            sample = np.load(gridinfo_path, allow_pickle=True)
            grids = sample['xyz']
            if (not self.static) and self.config.augmentation.randomize_grid > 1e-3:
                grids = self._apply_grid_randomization(grids)
                
            cats, mask = None, None
            if (not self.static) and self.config.cross_validation.motif_otf and 'PHcoord' in sample:
                cats = self._label_PH(grids, sample.get('PHcoord'), sample.get('PHtype'))
                
            elif 'labels' in sample or 'label' in sample:
                cats = sample.get('labels', sample.get('label'))
                # print (cats) 
            else:
                # return None # this was the original behavior, no cat samples just ignored (even for aff task)
                return grids, cats, mask # return grids with None cats, mask (this is desired?)
                #cats = np.zeros((len(grids), self.config.processing.ntype))
                
        except Exception as e:
            logger.error(f"Error loading grid data from {gridinfo_path}: {e}")
            return None

        if len(cats) > 0:
            if cats.shape[1] > self.config.processing.ntype:
                cats = cats[:, :self.config.processing.ntype]
            mask = np.sum(cats > 0, axis=1)
            cats = torch.tensor(cats).float()
            mask = torch.tensor(mask).float()

        return grids, cats, mask

    def _report_grid_and_motif(self, outf: str, grids: np.ndarray, cats: np.ndarray,
                               threshold: float = 0.01):
        out = open(outf,'w')
        nlabeled = 0
        form = "HETATM %4d  %2s  %2s  X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"
        for i,(l,grid) in enumerate(zip(cats,grids)):
            out.write(form%(i,'H','H',i,grid[0],grid[1],grid[2],0.0))
            
            if max(l) < threshold: continue
            for j in np.where(l>threshold)[0]:
                B = np.sqrt(l[j])
                nlabeled += 1
                mname = ['H','CB','CA','CD','CH','CR'][j]
                out.write(form%(i,mname,mname,i,grid[0],grid[1],grid[2],B))
        out.close()

    def _label_PH(self, grids, PHxyz, PHtypes, sig=1.0, out=None):
        indices_true, indices_fake = [],[]
        kd      = scipy.spatial.cKDTree(grids)
        kd_true   = scipy.spatial.cKDTree(PHxyz) # M x 3
        indices_true = np.concatenate(kd_true.query_ball_tree(kd, 1.5))
        indices_true = np.array(np.unique(indices_true),dtype=np.int16)

        # distance b/w grid & true-labeled-motifs
        dv2xyz = np.array([[g-x for g in grids[indices_true]] for x in PHxyz]) # grids x numTrue
        d2xyz = np.sum(dv2xyz*dv2xyz,axis=2)
        overlap = np.exp(-d2xyz/sig/sig) # Gaussian decay

        N = 6 #cats_true.shape[1] -- hard-coded
        label = np.zeros((len(grids),N))

        # assign labels
        for o,cat in zip(overlap,PHtypes): # motif index
            for j,p in enumerate(o): # grid index
                if p > 0.01:
                    label[indices_true[j],cat] = max(label[indices_true[j],cat],np.sqrt(p))

        # just for book-keeping
        if out != None:
            nlabeled = 0
            for i,l in enumerate(label):
                grid = grids[i]
                if max(l) > 0.01:
                    imotif = np.where(l>0.01)[0]
                    for j in imotif:
                        B = np.sqrt(l[j])
                        nlabeled += 1
                        mname = ['H','CB','CA','CD','CH','CR'][j]
                        out.write("HETATM %4d  %2s  %2s  X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,mname,mname,i,grid[0],grid[1],grid[2],B))
                out.write("HETATM %4d  H   H   X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,i,grid[0],grid[1],grid[2],0.0))
                
        return label
    
    def _apply_grid_randomization(self, grids: np.ndarray) -> np.ndarray:
        # uniform translation
        #rand_xyz = 2.0 * self.config.augmentation.randomize_grid * (0.5 - np.random.rand(len(grids), 3))
        rand_xyz = 2.0 * self.config.augmentation.randomize_grid * (0.5 - np.random.rand(3))
        return grids + rand_xyz

    def _process_ligands(self, target: str, ligands: List,
                        keyatoms_dict: Dict) -> Optional[Tuple]:
        try:
            ligands_parsed, mol2_type, data_type, is_near_native = self._parse_ligands(target, ligands)

            if ligands_parsed is None:
                return (None, False)

            if mol2_type == 'batch':
                ligand_data = self._load_batch_ligands(target, ligands_parsed, keyatoms_dict)
            elif mol2_type == 'single':
                ligand_data = self._load_single_ligands(target, ligands_parsed, keyatoms_dict, is_near_native)
            if ligand_data is None:
                return (None, False)

            ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_read = ligand_data
            if len(ligand_graphs) == 0:
                logger.error(f"No valid ligand graphs found for {target}")
                return (None, False)

            actives = [ligands_parsed[0]]
            # Strip source prefix if present (e.g., "biolip.xyz" -> "xyz")
            active_name = actives[0].split('.')[-1] if '.' in actives[0] else actives[0]

            # Critical check: active ligand MUST be successfully loaded
            # Without active, binding_labels would be all zeros, causing:
            # - BCE loss to train on wrong labels
            # - CE ranking loss to rank first decoy highest (harmful!)
            # - Corrupted Pt/Pf metrics (decoy added to true positives)
            if active_name not in tags_read:
                logger.warning(f"Active ligand '{active_name}' failed to load for {target}, skipping sample")
                return (None, False)

            binding_labels = [1 if tag == active_name else 0 for tag in tags_read]
            active_idx = tags_read.index(active_name)
            native_graph = native_graphs[active_idx] if native_graphs else None

            # Extract key_xyz from native graph
            native_com = None  # COM of native ligand for coordinate transform
            key_xyz = None
            if native_graph is not None and not isinstance(native_graph, list):
                key_indices = key_indices_list[active_idx]
                key_xyz = native_graph.ndata['x'][key_indices]
                # Get stored COM for coordinate frame transformation
                if hasattr(native_graph, 'com'):
                    native_com = native_graph.com  # Shape: [1, 3]

            # If key_xyz extraction failed, skip sample to avoid shape mismatch in str loss
            if key_xyz is None:
                logger.warning(f"Failed to extract key_xyz for active '{active_name}' in {target}, skipping sample")
                return (None, False)
            ligand_info = {
                'nK': [len(idx) for idx in key_indices_list],
                'ligands': tags_read,
                'atms': atoms_list,
                'native_com': native_com,  # For coordinate frame transformation
            }

            return (ligand_graphs, native_graph, key_xyz, key_indices_list,
                   binding_labels, ligand_info), is_near_native

        except Exception as e:
            logger.error(f"Error processing ligands for {target}: {e}")
            traceback.print_exc()
            return (None, False)

    def _parse_ligands(self, target: str, ligands: List) -> Tuple:
        source = target.split('/')[0]
        pname = target.split('/')[1]
        is_near_native = False
        if source in ['docked']:
            data_type = 'model'
        elif source in ['biolip','pdbbind']:
            data_type = 'structure'
        elif source in ['chembl','dude']:
            data_type = 'activity'

        (active_ligand, mol2_type) = ligands[0], ligands[1] if len(ligands) > 1 else 'single'
        if mol2_type == 'single':
            active = [pname]
            decoys = self.decoys.get(f'{source}.{pname}', [])
            active, is_near_native = self._get_active_or_near_native(active, source)
        elif mol2_type == 'batch':
            # mol2_file = f"{self.config.paths.datapath}/{source}/{pname}/{active_ligand}.mol2"
            batch_subdir = getattr(self.config.paths, 'chembl_batch_subdir', 'batch_mol2s')
            mol2_file = f"{self.config.paths.datapath}/{source}/{pname}/{batch_subdir}/{active_ligand}_b.mol2"
            if os.path.exists(mol2_file):
                active_and_decoys = myutils.read_mol2_batch(mol2_file, tag_only=True)[-1]
                active = [active_and_decoys[0]]
                decoys = active_and_decoys[1:]
            else:
                return None, None, None, None
        else:
            raise ValueError(f"Unknown mol2_type: {mol2_type}")

        n_decoy_slots = self.config.processing.max_subset - len(active)
        if len(decoys) > n_decoy_slots:
            target_key = f'{source}.{pname}'
            hard_ids = getattr(self, '_hard_negatives', {}).get(target_key, [])
            n_hard_slots = getattr(self, '_hard_neg_slots', 5)
            if hard_ids and not self.static:
                hard_in_pool = [d for d in hard_ids if d in decoys][:n_hard_slots]
                remaining = [d for d in decoys if d not in hard_in_pool]
                n_random = n_decoy_slots - len(hard_in_pool)
                random_fill = list(np.random.choice(remaining, min(n_random, len(remaining)), replace=False))
                decoys = hard_in_pool + random_fill
            else:
                decoys = list(np.random.choice(decoys, n_decoy_slots, replace=False))

        final_ligands = active + decoys
        #print (target, 'ligands:', final_ligands)
        return final_ligands, mol2_type, data_type, is_near_native

    def _get_active_or_near_native(self, active: List[str], source: str) -> List[str]:
        """
        Randomly select a near-native ligand from the cluster instead of the actual active.

        Args:
            active: List containing the active ligand name
            source: Data source (biolip, pdbbind) NO CHEMBL!!!

        Returns:
            List containing either the original active or a randomly selected near-native
        """
        is_near_native = False

        # Only apply near-native augmentation during training, not validation
        if self.static:
            return active, is_near_native

        if not self.crossactives or len(active) == 0:
            return active, is_near_native

        active_name = active[0]
        cluster_key = f"{source}.{active_name}"

        # Check if this active has near-native analogs in the clusters
        if cluster_key not in self.crossactives:
            return active, is_near_native

        cluster_members = self.crossactives[cluster_key]
        # Exclude self from near-native candidates
        near_natives = [m for m in cluster_members if m != cluster_key]
        if not near_natives:
            return active, is_near_native

        # Randomly decide whether to use near-native (configurable probability)
        near_native_prob = getattr(self.config.cross_validation, 'cross_val_prob', 0.0)
        if near_native_prob > 0 and np.random.rand() < near_native_prob:
            is_near_native = True
            # Randomly select one near-native ligand
            selected = np.random.choice(near_natives)
            # print(f"Using near-native {selected} instead of {active_name}")
            return [selected], is_near_native

        return active, is_near_native

    def _load_batch_ligands(self, target: str, ligands: List[str],
                          keyatoms_dict: Dict) -> Optional[Tuple]:
        source = target.split('/')[0]
        pname = target.split('/')[1]
        active_ligand = ligands[0]
        batch_subdir = getattr(self.config.paths, 'chembl_batch_subdir', 'batch_mol2s')
        mol2_file = f"{self.config.paths.datapath}/{source}/{pname}/{batch_subdir}/{active_ligand}_b.mol2"
        try:
            mol_data = self.loader.read_mol2_batch(mol2_file, ligands)
            if mol_data is None:
                return None

            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = mol_data

            ligand_graphs = []
            key_indices_list = []
            atoms_list = []
            tags_processed = []

            for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
                elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read
            ):
                try:
                    mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
                    graph = self.graph_builder.build_ligand_graph(mol_tuple, name=tag)

                    if graph is None:
                        continue

                    com = torch.mean(graph.ndata['x'], axis=0).float()
                    graph.ndata['x'] = (graph.ndata['x'] - com).float()
                    graph.com = com  # Store COM for coordinate transform (graph-level attribute)

                    if self.config.processing.drop_H:
                        filtered_atoms = [atom for atom, element in zip(atm, elem) if element != 'H']
                    else:
                        filtered_atoms = atm

                    key_indices = self._identify_key_atoms(tag, filtered_atoms, keyatoms_dict)

                    if not key_indices:
                        logger.warning(f"No key atoms found for {tag}")
                        continue

                    ligand_graphs.append(graph)
                    key_indices_list.append(key_indices)
                    atoms_list.append(filtered_atoms)
                    tags_processed.append(tag)

                except Exception as e:
                    logger.error(f"Error processing ligand {tag}: {e}")
                    continue

            native_graphs = ligand_graphs

            return ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_processed

        except Exception as e:
            logger.error(f"Error loading batch ligands from {mol2_file}: {e}")
            return None

    def _load_single_ligands(self, target: str, ligands: List[str], keyatoms_dict: Dict, is_near_native: bool) -> Optional[Tuple]:
        source = target.split('/')[0]
        pname = target.split('/')[1]
        try:
            ligand_graphs = []
            native_graphs = []
            key_indices_list = []
            atoms_list = []
            tags_read = []
            for i, ligand in enumerate(ligands):
                if i == 0:
                    if is_near_native:
                        source = ligand.split('.')[0]
                        pname = ligand.split('.')[1]
                        # Load keyatoms for the near-native ligand (not original receptor)
                        keyatomf = self.loader.find_keyatomf(pname, source)
                        if keyatomf:
                            keyatoms_dict = self.loader.load_keyatoms(keyatomf, targetname=pname)
                    if source == 'pdbbind':
                        mol2_path = f"{self.config.paths.datapath}/{source}/{pname}/{pname}_ligand.mol2"
                    elif source == 'biolip':
                        mol2_path = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.qh.mol2"
                    lig_pname = pname
                else:
                    lig_source = ligand.split('.')[0]
                    lig_pname = ligand.split('.')[1]
                    if lig_source == 'pdbbind':
                        mol2_path = f"{self.config.paths.datapath}/{lig_source}/{lig_pname}/{lig_pname}_ligand.mol2"
                    elif lig_source == 'biolip':
                        mol2_path = f"{self.config.paths.datapath}/{lig_source}/{lig_pname}/{lig_pname}.qh.mol2"
                    elif lig_source == 'zinc':
                        mol2_path = f"{self.config.paths.datapath}/{lig_source}/{lig_pname}/{lig_pname}.qh.mol2"
                    keyatomf = self.loader.find_keyatomf(lig_pname, lig_source)
                    if not keyatomf:
                        continue
                    keyatoms_dict = self.loader.load_keyatoms(keyatomf, targetname=lig_pname)
                try:
                    if mol2_path.endswith('.mol2'):
                        mol2_file = mol2_path
                    else:
                        mol2_file = os.path.join(mol2_path, f'{ligand}.ligand.mol2')

                    clean_ligand = lig_pname
                    mol_data = self.loader.read_mol2_single(mol2_file)
                    if mol_data is None:
                        continue

                    ligand_graph = self.graph_builder.build_ligand_graph(mol_data, name=clean_ligand)
                    if ligand_graph is None:
                        continue

                    com = torch.mean(ligand_graph.ndata['x'], axis=0).float()
                    ligand_graph.ndata['x'] = (ligand_graph.ndata['x'] - com).float()

                    native_graph = copy.deepcopy(ligand_graph)
                    native_graph.com = com  # Store COM for coordinate transform (graph-level attribute)

                    # Conformer augmentation: only for active ligands (i==0)
                    # Behavior depends on conformer_prob:
                    #   prob < 1.0: prob = P(conformer), sample from conformers only (not native)
                    #   prob = 1.0: uniform sample from [native, conf_0, conf_1, ...]
                    # Note: key_xyz (structure target) comes from native_graph (crystal coords)
                    #       ligand_graph gets sampled coords (model input only)
                    if i == 0 and self._should_use_conformer():
                        conf_mol2 = self._get_conformer_path(source, lig_pname)
                        if conf_mol2 and os.path.exists(conf_mol2):
                            try:
                                conf_xyz, _ = myutils.read_mol2s_xyzonly(conf_mol2)
                                if conf_xyz is not None and len(conf_xyz) > 0:
                                    n_crystal = ligand_graph.ndata['x'].shape[0]
                                    # Filter conformers with matching atom count
                                    valid_conf_xyz = [c for c in conf_xyz if c.shape[0] == n_crystal]

                                    if valid_conf_xyz:
                                        conf_prob = self.config.augmentation.conformer_prob

                                        if conf_prob >= 1.0:
                                            # prob=1.0: uniform over [native + conformers]
                                            native_xyz = ligand_graph.ndata['x'].squeeze(1).numpy()
                                            all_options = [native_xyz] + valid_conf_xyz
                                            selected_idx = np.random.randint(len(all_options))
                                            if selected_idx > 0:  # Conformer selected
                                                selected_xyz = all_options[selected_idx]
                                                ligand_graph.ndata['x'] = torch.tensor(selected_xyz).float()[:, None, :]
                                                ligand_graph.ndata['x'] -= ligand_graph.ndata['x'].mean(axis=0)
                                            # else: native stays
                                        else:
                                            # prob<1.0: sample from conformers only (native excluded)
                                            conf_idx = np.random.randint(len(valid_conf_xyz))
                                            ligand_graph.ndata['x'] = torch.tensor(valid_conf_xyz[conf_idx]).float()[:, None, :]
                                            ligand_graph.ndata['x'] -= ligand_graph.ndata['x'].mean(axis=0)
                                    else:
                                        logger.warning(f"No valid conformers for {lig_pname}: all have atom count mismatch (crystal={n_crystal})")
                            except Exception as e:
                                logger.debug(f"Failed to load conformer from {conf_mol2}: {e}")

                    atoms = mol_data[6]
                    key_indices = self._identify_key_atoms(clean_ligand, atoms, keyatoms_dict)
                    if not key_indices:
                        continue

                    ligand_graphs.append(ligand_graph)
                    native_graphs.append(native_graph)
                    key_indices_list.append(key_indices)
                    atoms_list.append(atoms)
                    tags_read.append(clean_ligand)

                except Exception as e:
                    logger.error(f"Error processing single ligand {ligand}: {e}")
                    continue
            return ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_read

        except Exception as e:
            logger.error(f"Error loading single ligands: {e}")
            return None

    def _identify_key_atoms(self, target: str, atoms: List[str], keyatoms_dict: Dict) -> List[int]:
        if target not in keyatoms_dict:
            return []
        # print (target, keyatoms_dict[target])
        key_indices = [atoms.index(atom) for atom in keyatoms_dict[target] if atom in atoms]
        if len(key_indices) > 10:
            key_indices = list(np.random.choice(key_indices, 10, replace=False))
        return key_indices

    def _build_info_dict(self, pname: str, target: str, origin: torch.Tensor,
                        grids: np.ndarray, grid_indices: np.ndarray, gridinfo: str,
                        ligand_info: Dict, is_near_native: bool) -> Dict:
        source = target.split('/')[0]
        if source in ['docked', 'biolip', 'pdbbind']:
            eval_struct = 1
        elif source in ['chembl','dude']:
            eval_struct = 0

        # For near-natives, check cross_eval_struct config
        if is_near_native:
            if getattr(self.config.cross_validation, 'cross_eval_struct', False):
                eval_struct = 1  # Calculate structure loss for near-natives
            else:
                eval_struct = 0

        info = {
            'pname': pname,
            'name': target,
            'com': origin,
            'grididx': grid_indices,
            'grid': grids,
            'gridinfo': gridinfo,
            'source': source,
            'eval_struct': eval_struct,
            'is_near_native': is_near_native  # Pass to training for weight adjustment
        }
        info.update(ligand_info)
        return info

    def _load_decoys_npz(self) -> Dict:
        decoys = {}
        file_path = self.config.paths.decoyf
        if os.path.exists(file_path):
            logger.info(f"Loading decoy NPZ: {file_path}")
            decoys = np.load(file_path, allow_pickle=True)['decoys'].item()
        return decoys
    
    def _load_decoys(self) -> Dict:
        target_decoy_dict = {}
        if self.config.paths.decoyf.endswith('.npz'):
            target_decoy_dict = self._load_decoys_npz()
        else:
            file_path = self.config.paths.decoyf
            if os.path.exists(file_path):
                logger.info(f"Loading decoy file: {file_path}")
                try:
                    with open(file_path, 'rb') as f:
                        decoy_map = pickle.load(f)
                    for target, decoys in decoy_map.items():
                        target_decoy_dict[target] = decoys
                except Exception as e:
                    logger.error(f"Error loading decoy file {file_path}: {e}")
        return target_decoy_dict

    def _load_crossactives(self) -> Dict:
        """
        Load cross-receptor clusters and create member -> cluster_members mapping.

        Returns dict where ANY member can look up all members in its cluster,
        not just representatives.
        """
        cross_file = self.config.paths.crossreceptorf

        # First pass: build rep -> members mapping
        rep_to_members = {}
        with open(cross_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    rep, mem = parts
                    if rep not in rep_to_members:
                        rep_to_members[rep] = []
                    rep_to_members[rep].append(mem)

        # Second pass: create member -> cluster_members mapping for ALL members
        member_to_cluster = {}
        for rep, members in rep_to_members.items():
            for mem in members:
                member_to_cluster[mem] = members

        logger.info(f"Loaded {len(rep_to_members)} clusters, {len(member_to_cluster)} member mappings")
        return member_to_cluster

    def _should_use_conformer(self) -> bool:
        """
        Determine whether to use conformer augmentation for this sample.

        Returns True if:
        - use_conformers is enabled in config
        - Not in validation mode (static=True), OR conformer_validation is enabled
        - Random probability check passes
        """
        aug_config = self.config.augmentation

        # Check if conformer augmentation is enabled
        if not aug_config.use_conformers:
            return False

        # Check validation mode
        if self.static and not aug_config.conformer_validation:
            return False

        # Probability check
        if np.random.rand() >= aug_config.conformer_prob:
            return False

        return True

    def _get_conformer_path(self, source: str, pname: str) -> Optional[str]:
        """
        Build the conformer file path for a given target.

        Args:
            source: Data source (pdbbind, biolip)
            pname: Target name

        Returns:
            Path to conformer mol2 file, or None if not applicable
        """
        suffix = self.config.augmentation.conformer_suffix
        base_path = f"{self.config.paths.datapath}/{source}/{pname}"
        conf_path = f"{base_path}/{pname}{suffix}"
        return conf_path

    def _get_null_result(self, pname: str) -> Tuple:
        info = {'pname': pname}
        return (None, None, None, None, None, None, None, info)

def collate(samples):
    valid_samples = [s for s in samples if s is not None and s[0] is not None]
    if not valid_samples:
        return None
    receptor_graphs = [s[0] for s in valid_samples]
    ligand_graphs = [s[1] for s in valid_samples]
    cats = [s[2] for s in valid_samples]
    masks = [s[3] for s in valid_samples]
    key_xyz = [s[4] for s in valid_samples]
    key_indices = [s[5] for s in valid_samples]
    binding_labels = [s[6] for s in valid_samples]
    info_dicts = [s[-1] for s in valid_samples]
    batched_receptors = dgl.batch(receptor_graphs)
    if None in cats:
        batched_cats = None
        batched_masks = None
    else:
        batched_cats = torch.stack(cats, dim=0).squeeze()
        if len(batched_cats.shape) == 2:
            batched_cats = batched_cats[None, :]
        batched_masks = torch.stack(masks, dim=0).float().squeeze()
        if len(batched_masks.shape) == 1:
            batched_masks = batched_masks[None, :]
    combined_info = {}
    for key in info_dicts[0]:
        combined_info[key] = [info_dict[key] for info_dict in info_dicts]
    combined_info['grid'] = torch.tensor(np.array(combined_info['grid']))
    batched_grid_indices = []
    node_offset = 0
    for num_nodes, grid_idx in zip(batched_receptors.batch_num_nodes(), combined_info['grididx']):
        batched_grid_indices.append(torch.tensor(grid_idx, dtype=int) + node_offset)
        node_offset += num_nodes
    combined_info['grididx'] = torch.cat(batched_grid_indices, dim=0)
    if ligand_graphs[0]:
        batched_ligands = dgl.batch(ligand_graphs[0])
        global_data = torch.stack([g.gdata for g in ligand_graphs[0]])
        setattr(batched_ligands, "gdata", global_data)
        try:
            batched_key_xyz = key_xyz[0].squeeze()[None, :, :]
            batched_key_indices = [torch.eye(n)[idx] for n, idx in zip(batched_ligands.batch_num_nodes(), key_indices[0])]
            batched_binding_labels = torch.tensor(binding_labels[0], dtype=float)
            combined_info['nK'] = torch.tensor(combined_info['nK'][0])
        except Exception as e:
            batched_ligands = None
            batched_key_xyz = torch.tensor(0.0)
            batched_key_indices = torch.tensor(0.0)
            batched_binding_labels = torch.tensor([])
    else:
        batched_ligands = None
        batched_key_xyz = torch.tensor(0.0)
        batched_key_indices = torch.tensor(0.0)
        batched_binding_labels = torch.tensor([])

    return (batched_receptors, batched_ligands, batched_cats, batched_masks,
           batched_key_xyz, batched_key_indices, batched_binding_labels, combined_info)

# Backward compatibility alias
DataSet = TrainingDataSet
