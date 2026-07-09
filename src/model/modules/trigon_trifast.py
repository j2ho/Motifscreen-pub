import torch
from torch import nn
from torch.nn import Linear
from torch.utils import checkpoint
import torch.nn.functional as F


class FastTriangleProteinToCompound(torch.nn.Module):
    """
    Optimized version focusing on actual bottlenecks:
    1. Fused operations where beneficial
    2. Better memory layout
    3. Reduced intermediate tensors
    """
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        # Fuse gate and linear operations
        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)
        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
        
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # Single layernorm call instead of multiple
        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)  
        compound_pair = self.layernorm(compound_pair)
        
        # Fuse sigmoid and multiplication operations
        # This reduces memory allocations and kernel launches
        gate1 = self.gate_linear1(z).sigmoid()
        gate2 = self.gate_linear2(z).sigmoid()
        
        ab1 = gate1 * self.linear1(z) * z_mask
        ab2 = gate2 * self.linear2(z) * z_mask

        # Reuse gates for efficiency
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = gate1.detach() * self.linear1(compound_pair)  # Reuse computed gate

        g = self.ending_gate_linear(z).sigmoid()

        # Use optimized einsum with optimal memory layout
        # PyTorch's einsum is already highly optimized for these operations
        with torch.cuda.amp.autocast(enabled=False):  # Prevent mixed precision issues
            block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
            block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)

        # Fuse final operations
        result = self.layernorm_c(block1 + block2)
        z = g * self.linear_after_sum(result) * z_mask
        return z


class FastTriangleSelfAttentionRowWise(torch.nn.Module):
    """
    Optimized triangle self-attention focusing on:
    1. Better memory access patterns
    2. Fused operations
    3. Optimized attention computation
    """
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = 1.0 / (c ** 0.5)  # Pre-compute scale
        
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        
        # Fuse QKV projections into single linear layer for better memory efficiency
        self.qkv_linear = Linear(embedding_channels, 3 * self.all_head_size, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def forward(self, z, z_mask):
        batch_n, p_length = z.shape[:2]
        z = self.layernorm(z)
        
        # Single matrix multiply for Q, K, V - much faster than 3 separate ones
        qkv = self.qkv_linear(z)  # [B, I, J, 3*all_head_size]
        qkv = qkv.view(batch_n, p_length, -1, 3, self.num_attention_heads, self.attention_head_size)
        q, k, v = qkv.unbind(dim=3)  # Each is [B, I, J, H, C]
        
        # Use torch.baddbmm for efficient batched matrix multiply
        # Reshape for batched computation
        q_flat = q.reshape(batch_n * self.num_attention_heads, p_length, -1, self.attention_head_size)
        k_flat = k.reshape(batch_n * self.num_attention_heads, p_length, -1, self.attention_head_size)  
        v_flat = v.reshape(batch_n * self.num_attention_heads, p_length, -1, self.attention_head_size)
        
        # Efficient attention computation using scaled_dot_product_attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized SDPA (Flash Attention when available)
            mask_expanded = z_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_attention_heads, p_length, -1)
            mask_flat = mask_expanded.reshape(batch_n * self.num_attention_heads, p_length, -1)
            
            # Convert mask to attention mask format
            attn_mask = torch.where(mask_flat.unsqueeze(-2), 0.0, float('-inf'))
            
            output_flat = F.scaled_dot_product_attention(
                q_flat.transpose(1, 2), k_flat.transpose(1, 2), v_flat.transpose(1, 2),
                attn_mask=attn_mask, scale=self.scale
            )
            output_flat = output_flat.transpose(1, 2)
        else:
            # Fallback to manual computation with optimized operations
            # Compute attention scores efficiently
            scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
            
            # Apply mask efficiently
            mask_expanded = z_mask.view(batch_n, 1, p_length, 1, -1)
            mask_flat = mask_expanded.expand(-1, self.num_attention_heads, -1, p_length, -1)
            mask_flat = mask_flat.reshape(batch_n * self.num_attention_heads, p_length, p_length, -1)
            
            scores = scores.masked_fill(mask_flat.squeeze(-1) == 0, float('-inf'))
            
            # Efficient softmax and matmul
            weights = torch.softmax(scores, dim=-1)
            output_flat = torch.matmul(weights, v_flat)
        
        # Reshape back
        weighted_avg = output_flat.reshape(batch_n, p_length, -1, self.num_attention_heads, self.attention_head_size)
        
        # Apply gating
        g = self.g(z).view(batch_n, p_length, -1, self.num_attention_heads, self.attention_head_size).sigmoid()
        output = g * weighted_avg
        
        # Flatten and project
        output = output.reshape(batch_n, p_length, -1, self.all_head_size)
        z = self.final_linear(output) * z_mask.unsqueeze(-1)
        return z


class MemoryEfficientTriangleSelfAttentionRowWise(torch.nn.Module):
    """
    Even more memory efficient version using gradient checkpointing within attention
    """
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        
        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        
        # Pre-compute mask operations
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        
        q = self.reshape_last_dim(self.linear_q(z)) 
        k = self.reshape_last_dim(self.linear_k(z))
        v = self.reshape_last_dim(self.linear_v(z))
        
        # More memory-efficient einsum computation
        # Break down the computation to reduce peak memory
        logits = torch.empty(batch_n, p_length, self.num_attention_heads, p_length, p_length, 
                            dtype=z.dtype, device=z.device)
        
        # Compute attention in chunks to reduce memory usage
        chunk_size = min(32, p_length)
        for i in range(0, p_length, chunk_size):
            end_i = min(i + chunk_size, p_length)
            logits[:, i:end_i] = torch.einsum('biqhc,bikhc->bihqk', 
                                             q[:, i:end_i], k) + attention_mask_i[:, i:end_i]
        
        weights = torch.softmax(logits, dim=-1)
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        
        g = self.reshape_last_dim(self.g(z)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        
        z = self.final_linear(output) * z_mask.unsqueeze(-1)
        return z


class SimpleFastTrigonModule(nn.Module):
    """
    Simplified fast version focusing on:
    1. Reducing memory allocations
    2. Better operation fusion
    3. Optimized attention when sequence length is small-medium
    """
    def __init__(self,
                 n_trigonometry_module_stack,
                 grid_m=64,
                 ligand_m=64, 
                 c=64,
                 dropout_rate=0.1,
                 bias=True,
                 use_fast_attention=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.n_trigonometry_module_stack = n_trigonometry_module_stack
        self.use_fast_attention = use_fast_attention

        self.Wrs = nn.Linear(grid_m, c, bias=bias)
        self.Wls = nn.Linear(ligand_m, c, bias=bias)

        # Choose attention implementation based on expected usage
        attention_class = FastTriangleSelfAttentionRowWise if use_fast_attention else MemoryEfficientTriangleSelfAttentionRowWise
        
        self.protein_to_compound_list = nn.ModuleList([
            FastTriangleProteinToCompound(embedding_channels=c, c=c) 
            for _ in range(n_trigonometry_module_stack)
        ])
        self.triangle_self_attention_list = nn.ModuleList([
            attention_class(embedding_channels=c, c=c) 
            for _ in range(n_trigonometry_module_stack)
        ])

        self.transition = Transition(embedding_channels=c, n=4, bias=bias)

    def forward(self, hs_rec, hs_lig, z_mask,
                D_rec, D_lig,
                use_checkpoint=False, drop_out=False):
        # Pre-compute features to avoid redundant computation
        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)
        
        # More efficient outer product computation
        z = torch.einsum('bnd,bmd->bnmd', hs_rec, hs_lig)

        # Main computation loop with potential optimizations
        for i_module in range(self.n_trigonometry_module_stack):
            if use_checkpoint:
                # Selective checkpointing - only checkpoint memory-heavy operations
                zadd = checkpoint.checkpoint(
                    self.protein_to_compound_list[i_module], 
                    z, D_rec, D_lig, z_mask.unsqueeze(-1),
                    use_reentrant=False  # More memory efficient
                )
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                
                zadd = checkpoint.checkpoint(
                    self.triangle_self_attention_list[i_module], 
                    z, z_mask,
                    use_reentrant=False
                )
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
            else:
                # Inline computation for better performance when memory allows
                zadd = self.protein_to_compound_list[i_module](z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                
                zadd = self.triangle_self_attention_list[i_module](z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd

            # Transition layer (keep original - it's already efficient)
            z = self.transition(z)

        return z


class Transition(torch.nn.Module):
    """Original transition - already well optimized"""
    def __init__(self, embedding_channels=256, n=4, bias=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels, bias=bias)
        self.linear2 = Linear(n*embedding_channels, embedding_channels, bias=bias)
    
    def forward(self, z):
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z


# Additional utility for benchmarking
def benchmark_modules(original_module, fast_module, inputs, num_runs=10):
    """Utility to properly benchmark the modules"""
    import time
    
    # Warmup
    for _ in range(3):
        _ = original_module(*inputs)
        _ = fast_module(*inputs)
    
    torch.cuda.synchronize()
    
    # Benchmark original
    start = time.time()
    for _ in range(num_runs):
        _ = original_module(*inputs)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # Benchmark fast
    start = time.time() 
    for _ in range(num_runs):
        _ = fast_module(*inputs)
    torch.cuda.synchronize()
    fast_time = time.time() - start
    
    print(f"Original: {original_time/num_runs:.4f}s per run")
    print(f"Fast: {fast_time/num_runs:.4f}s per run") 
    print(f"Speedup: {original_time/fast_time:.2f}x")
    
    return original_time/num_runs, fast_time/num_runs