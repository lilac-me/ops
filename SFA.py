# Copyright (c) 2023-2025; Songlin Yang; Yu Zhang
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

LOG2_E: tl.constexpr = 1.4426950408889634


@dataclass(frozen=True)
class TilingBlockConfig:
    BLOCK_N: int
    BLOCK_H: int
    BLOCK_Q_BWD: int
    BLOCK_K_BWD: int
    BLOCK_H_BWD: int
    extra_args: dict


CONFIG_MAP = {
    128: TilingBlockConfig(
        BLOCK_N=64, BLOCK_H=32, BLOCK_Q_BWD=64, BLOCK_K_BWD=64, BLOCK_H_BWD=32,
        extra_args={
            "multibuffer": True,
            "limit_auto_multi_buffer_only_for_local_buffer": False,
            "set_workspace_multibuffer": 4,
            "tile_mix_vector_loop": 2,
            "tile_mix_cube_loop": 2,
            }
    ),
    160: TilingBlockConfig(
        BLOCK_N=80, BLOCK_H=32, BLOCK_Q_BWD=80, BLOCK_K_BWD=40, BLOCK_H_BWD=32,
        extra_args={
            "multibuffer": True,
            "limit_auto_multi_buffer_only_for_local_buffer": False,
            "set_workspace_multibuffer": 4,
            "tile_mix_vector_loop": 2,
            "tile_mix_cube_loop": 2,
            }
    ),
    640: TilingBlockConfig(
        BLOCK_N=80, BLOCK_H=32, BLOCK_Q_BWD=80, BLOCK_K_BWD=64, BLOCK_H_BWD=32,
        extra_args={
            "multibuffer": True,
            "limit_auto_multi_buffer_only_for_local_buffer": False,
            "set_workspace_multibuffer": 4,
            "tile_mix_vector_loop": 2,
            "tile_mix_cube_loop": 2,
            }
    ),
}


class SparseFlashAttentionTriton(torch.autograd.Function):
    """
    Custom Autograd Function for Sparse Flash Attention with Discrete KV and SBHD layout.
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """
        Computes sparse attention forward pass.

        Args:
            q: Query tensor. Shape [Seq, Batch, Head, Dim].
            kv: Key-Value storage. Shape [Seq_kv, Batch, Dim].
            attn_sink: Attention sink tensor for stability. Shape [Head].
            topk_idxs: Sparse indices mapping queries to KV blocks. Shape [Seq, Batch, TopK].
            sm_scale: Softmax scaling factor.

        Returns:
            out: Attention output. Shape [Seq, Batch, Head, Dim].
        """
        
        n_ctx, batch, n_heads, head_dim = q.shape
        kv_ctx =  kv.shape[0]

        # Config lookup for Triton tiling parameters
        topk = topk_idxs.shape[-1]
        if topk not in CONFIG_MAP:
            raise ValueError(f"Unsupported topk value: {topk}. Please add it to CONFIG_MAP.")
        cfg = CONFIG_MAP[topk]

        out = torch.empty_like(q)

        # LSE is required for numerical stability and backward pass re-computation
        log_sum_exp = torch.empty(
            (n_ctx, batch, n_heads),
            device=q.device,
            dtype=torch.float32
        )

        grid = (batch, n_ctx)

        # Intermediate buffers for Triton kernel execution
        # NOTE: V_Buffer removed - KV shares storage, only one buffer needed
        K_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_N, head_dim), device=q.device, dtype=torch.bfloat16)
        QK_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H, cfg.BLOCK_N), device=q.device, dtype=torch.float32)
        P_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H, cfg.BLOCK_N), device=q.device, dtype=torch.bfloat16)
        PV_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H, head_dim), device=q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q_ptr=q,
            KV_ptr=kv,
            TopKIdx_ptr=topk_idxs,
            Sink_ptr=attn_sink,
            LSE_ptr=log_sum_exp,
            Out_ptr=out,
            K_Buffer_ptr=K_Buffer,
            QK_Buffer_ptr=QK_Buffer,
            P_Buffer_ptr=P_Buffer,
            PV_Buffer_ptr=PV_Buffer,
            sm_scale=sm_scale,
            stride_qz=q.stride(1),
            stride_qs=q.stride(0),
            stride_qh=q.stride(2),
            stride_qd=q.stride(3),
            stride_kvz=kv.stride(1),
            stride_kvs=kv.stride(0),
            stride_kvd=kv.stride(2),
            stride_iz=topk_idxs.stride(1),
            stride_is=topk_idxs.stride(0),
            stride_in=topk_idxs.stride(2),
            stride_oz=out.stride(1),
            stride_os=out.stride(0),
            stride_oh=out.stride(2),
            stride_od=out.stride(3),
            stride_sink=attn_sink.stride(0),
            stride_mz=log_sum_exp.stride(1),
            stride_ms=log_sum_exp.stride(0),
            stride_mh=log_sum_exp.stride(2),
            H=n_heads,
            HEAD_DIM=head_dim,
            BLOCK_N=cfg.BLOCK_N,
            TOPK=topk,
            BLOCK_H=cfg.BLOCK_H,
            KV_CTX=kv_ctx,
            **cfg.extra_args,
        )

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out, log_sum_exp)
        ctx.sm_scale = sm_scale
        ctx.kv_ctx = kv_ctx
        return out
    
    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ):
        q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors
        softmax_scale = ctx.sm_scale
        kv_ctx = ctx.kv_ctx

        n_ctx, batch, n_heads, head_dim = q.shape

        topk = int(topk_idxs.shape[-1])
        if topk not in CONFIG_MAP:
            raise ValueError(f"Unsupported topk value: {topk}. Please add it to CONFIG_MAP.")
        cfg = CONFIG_MAP[topk]

        if softmax_scale is None:
            softmax_scale = (1.0 / head_dim) ** 0.5

        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_kv = torch.zeros_like(kv, dtype=torch.float32)
        grad_sink = torch.zeros_like(attn_sink, dtype=torch.float32)

        K_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_Q_BWD, head_dim), device=q.device, dtype=torch.bfloat16)

        S_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_Q_BWD), device=q.device, dtype=torch.float32)
        dP_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_Q_BWD), device=q.device, dtype=torch.float32)
        dS_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_Q_BWD), device=q.device, dtype=torch.bfloat16)

        dQ_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, head_dim), device=q.device, dtype=torch.float32)

        num_blocks_q = triton.cdiv(topk, cfg.BLOCK_Q_BWD)
        grid = (batch, n_ctx)

        _attn_bwd_dq_dsink[grid](
            Q_ptr=q,
            KV_ptr=kv,
            Sink_ptr=attn_sink,
            TopKIdx_ptr=topk_idxs,
            grad_out_ptr=grad_out,
            grad_q_ptr=grad_q,
            grad_sink_ptr=grad_sink,
            LSE_ptr=lse,
            Out_ptr=out,
            k_buf_ptr=K_Buffer,
            s_buf_ptr=S_Buffer,
            dp_buf_ptr=dP_Buffer,
            ds_buf_ptr=dS_Buffer,
            dq_buf_ptr=dQ_Buffer,
            stride_qb=q.stride(1),
            stride_qm=q.stride(0),
            stride_qh=q.stride(2),
            stride_qd=q.stride(3),
            stride_kvb=kv.stride(1),
            stride_kvn=kv.stride(0),
            stride_kvd=kv.stride(2),
            stride_gob=grad_out.stride(1),
            stride_gom=grad_out.stride(0),
            stride_goh=grad_out.stride(2),
            stride_god=grad_out.stride(3),
            stride_gqb=grad_q.stride(1),
            stride_gqm=grad_q.stride(0),
            stride_gqh=grad_q.stride(2),
            stride_gqd=grad_q.stride(3),
            stride_tb=topk_idxs.stride(1),
            stride_tm=topk_idxs.stride(0),
            stride_tk=topk_idxs.stride(2),
            stride_lseb=lse.stride(1),
            stride_lsem=lse.stride(0),
            stride_lseh=lse.stride(2),
            stride_ob=out.stride(1),
            stride_om=out.stride(0),
            stride_oh=out.stride(2),
            stride_od=out.stride(3),
            stride_sink=attn_sink.stride(0),
            stride_gsink=grad_sink.stride(0),
            sm_scale=softmax_scale,
            TOPK=topk,
            n_ctx=n_ctx,
            n_heads=n_heads,
            head_dim=head_dim,
            BLOCK_K=cfg.BLOCK_Q_BWD,
            NUM_BLOCKS=num_blocks_q,
            BLOCK_H=cfg.BLOCK_H_BWD,
            KV_CTX=kv_ctx,
            **cfg.extra_args,
        )

        K_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_K_BWD, head_dim), device=q.device, dtype=torch.bfloat16)

        S_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_K_BWD), device=q.device, dtype=torch.float32)
        dP_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_K_BWD), device=q.device, dtype=torch.float32)
        P_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_K_BWD), device=q.device, dtype=torch.bfloat16)
        dS_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_H_BWD, cfg.BLOCK_K_BWD), device=q.device, dtype=torch.bfloat16)

        dK_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_K_BWD, head_dim), device=q.device, dtype=torch.float32)
        dV_Buffer = torch.empty((batch, n_ctx, cfg.BLOCK_K_BWD, head_dim), device=q.device, dtype=torch.float32)

        num_blocks_kv = triton.cdiv(topk, cfg.BLOCK_K_BWD)
        grid = (batch, n_ctx)

        _attn_bwd_dk_dv[grid](
            Q_ptr=q, 
            KV_ptr=kv, 
            TopKIdx_ptr=topk_idxs, 
            grad_out_ptr=grad_out,
            grad_kv_ptr=grad_kv, 
            LSE_ptr=lse, 
            Out_ptr=out,
            k_buf_ptr=K_Buffer, 
            s_buf_ptr=S_Buffer, 
            dp_buf_ptr=dP_Buffer,
            p_buf_ptr=P_Buffer, 
            ds_buf_ptr=dS_Buffer,
            dk_buf_ptr=dK_Buffer, 
            dv_buf_ptr=dV_Buffer,
            stride_qb=q.stride(1), stride_qm=q.stride(0), stride_qh=q.stride(2), stride_qd=q.stride(3),
            stride_kvb=kv.stride(1), stride_kvn=kv.stride(0), stride_kvd=kv.stride(2),
            stride_gob=grad_out.stride(1), stride_gom=grad_out.stride(0), stride_goh=grad_out.stride(2), stride_god=grad_out.stride(3),
            stride_gkvs=grad_kv.stride(0), stride_gkvd=grad_kv.stride(2),
            stride_tb=topk_idxs.stride(1), stride_tm=topk_idxs.stride(0), stride_tk=topk_idxs.stride(2),
            stride_lseb=lse.stride(1), stride_lsem=lse.stride(0), stride_lseh=lse.stride(2),
            stride_ob=out.stride(1), stride_om=out.stride(0), stride_oh=out.stride(2), stride_od=out.stride(3),
            sm_scale=softmax_scale,
            TOPK=topk,
            n_ctx=n_ctx,
            n_heads=n_heads,
            head_dim=head_dim,
            BLOCK_K=cfg.BLOCK_K_BWD,
            NUM_BLOCKS=num_blocks_kv,
            BLOCK_H=cfg.BLOCK_H_BWD,
            KV_CTX=kv_ctx,
            **cfg.extra_args,
        )

        return (
            grad_q, 
            grad_kv,
            grad_sink, 
            None, 
            None, 
        )


@triton.jit
def _inner_fwd(
    q_base,
    kv_base,
    idx_base,
    Sink_ptr,
    lse_base,
    out_base,
    k_buf_ptr,
    qk_buf_ptr,
    p_buf_ptr,
    pv_buf_ptr,
    stride_qh,
    stride_qd,
    stride_kvs,
    stride_kvd,
    stride_in,
    stride_sink,
    stride_mh,
    stride_oh,
    stride_od,
    off_d,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    START_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    """
    Inner loop for Ascend NPU Sparse FlashAttention utilizing Cube/Vector Mix Mode.

    Pipeline Workflow:
    -----------------
    The kernel implements a 5-stage pipeline to overlap memory access and computation:
    1. [Vector Core] Load KV: Fetches sparse KV blocks based on topk_idxs.
    2. [Cube Core]   Compute QK: Performs matrix multiplication (Q @ K.T).
    3. [Vector Core] Softmax: Computes online Max/Sum for numerical stability.
    4. [Cube Core]   Compute PV: Performs matrix multiplication (P @ V).
    5. [Vector Core] Accumulate: Updates the output running sum and LSE.

    Hardware Synchronization:
    ------------------------
    Uses `al.sync_block_set` and `al.sync_block_wait` to manage dependencies 
    between Vector and Cube execution units.
    """
    
    # --- Initialization & Head Partitioning ---
    num_steps = triton.cdiv(TOPK, BLOCK_N)
    off_h_full = START_H + tl.arange(0, BLOCK_H)
    h_mask_full = off_h_full < H

    # Load Query into register
    q_ptrs = q_base + off_h_full[:, None] * stride_qh + off_d[None, :] * stride_qd
    q_full = tl.load(q_ptrs, mask=h_mask_full[:, None], other=0.0)

    # Vector Core Sub-partitioning (1:2 Split)
    # sub_id identifies which of the two Vector sub-tasks is running
    sub_id = al.sub_vec_id()
    HALF_H: tl.constexpr = BLOCK_H // 2
    row_indices = tl.arange(0, HALF_H) + sub_id * HALF_H 
    off_h_sub = START_H + row_indices
    h_mask_sub = off_h_sub < H

    HALF_N: tl.constexpr = BLOCK_N // 2
    n_offset_local = sub_id * HALF_N
    n_offset_local1 = sub_id.to(tl.float32) * HALF_N
    
    # Online Softmax Accumulators
    acc = tl.zeros([HALF_H, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([HALF_H], -10e10, dtype=tl.float32)
    l_i = tl.zeros([HALF_H], dtype=tl.float32)

    off_n_buf = tl.arange(0, BLOCK_N)

    for i in range(num_steps):
        # 1. [Vector] Load Sparse KV + Cache full-block index
        start_n = i * BLOCK_N
        start_n1 = i.to(tl.float32) * BLOCK_N

        # Load full-block index once (reused in softmax masking stage)
        off_n_full_global0 = start_n + off_n_buf
        off_n_full_global1 = start_n1 + off_n_buf.to(tl.float32)
        n_mask_full = off_n_full_global1 < TOPK
        k_idx_full = tl.load(idx_base + off_n_full_global0 * stride_in, mask=n_mask_full, other=-1)
        mask_full = (k_idx_full.to(tl.float32) >= 0) & n_mask_full

        # Sub-partition load for Vector Core (each sub loads HALF_N)
        off_n_local = start_n + n_offset_local + tl.arange(0, HALF_N)
        off_n_local1 = start_n1 + n_offset_local1 + tl.arange(0, HALF_N).to(tl.float32)
        n_mask = off_n_local1 < TOPK

        k_idx = tl.load(idx_base + off_n_local * stride_in, mask=n_mask, other=-1)
        load_mask = (k_idx.to(tl.float32) >= 0) & n_mask

        # address conflict
        dummy_idx = (KV_CTX - 1) - tl.arange(0, HALF_N)

        k_idx_optimized = tl.where(load_mask, k_idx, dummy_idx)
        kv_ptrs = kv_base + k_idx_optimized[:, None] * stride_kvs + off_d[None, :] * stride_kvd

        kv = tl.load(kv_ptrs, mask=load_mask[:, None], other=0.0)

        buf_row_idx = n_offset_local + tl.arange(0, HALF_N)
        tl.store(k_buf_ptr + buf_row_idx[:, None] * HEAD_DIM + off_d[None, :], kv)

        # Signal 0: Sync Vector (Load) to Cube (Compute)
        al.sync_block_set("vector", "cube", 0)

        # 2. [Cube] Matrix Multi: Q @ K.T
        al.sync_block_wait("vector", "cube", 0)

        kv_load = tl.load(k_buf_ptr + off_n_buf[:, None] * HEAD_DIM + off_d[None, :])

        qk_full = tl.dot(q_full, tl.trans(kv_load))

        qk_store_ptr = qk_buf_ptr + tl.arange(0, BLOCK_H)[:, None] * BLOCK_N + off_n_buf[None, :]
        tl.store(qk_store_ptr, qk_full)

        # 3. [Vector] Softmax Update (m_i, l_i, p_sub)
        # Reuse cached mask_full from stage 1 — no redundant index load
        qk_load_ptr = qk_buf_ptr + row_indices[:, None] * BLOCK_N + off_n_buf[None, :]
        qk_sub = tl.load(qk_load_ptr)

        qk_sub *= (sm_scale * LOG2_E)
        qk_sub = tl.where(mask_full[None, :], qk_sub, -10e10)

        m_ij = tl.max(qk_sub, 1)
        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_next)
        p_sub = tl.math.exp2(qk_sub - m_next[:, None])

        acc = acc * alpha[:, None]

        p_store_ptr = p_buf_ptr + row_indices[:, None] * BLOCK_N + off_n_buf[None, :]
        tl.store(p_store_ptr, p_sub.to(tl.float16))

        # Signal 2: Sync Vector (Softmax) to Cube (PV)
        al.sync_block_set("vector", "cube", 2)

        # 4. [Cube] Matrix Multi: P @ V
        al.sync_block_wait("vector", "cube", 2)

        p_full = tl.load(p_buf_ptr + tl.arange(0, BLOCK_H)[:, None] * BLOCK_N + off_n_buf[None, :])

        pv_full = tl.dot(p_full.to(kv_load.dtype), kv_load)

        pv_store_ptr = pv_buf_ptr + tl.arange(0, BLOCK_H)[:, None] * HEAD_DIM + off_d[None, :]
        tl.store(pv_store_ptr, pv_full)

        # 5. [Vector] Result Accumulation
        pv_load_ptr = pv_buf_ptr + row_indices[:, None] * HEAD_DIM + off_d[None, :]
        pv_sub = tl.load(pv_load_ptr)

        acc += pv_sub
        l_i = l_i * alpha + tl.sum(p_sub, 1)
        m_i = m_next

    sink_ptrs = Sink_ptr + off_h_sub * stride_sink
    attn_sink = tl.load(sink_ptrs, mask=h_mask_sub, other=0.0) * LOG2_E

    p_sink = tl.math.exp2(attn_sink - m_i)
    l_i += p_sink
    lse = m_i + tl.math.log2(l_i)

    tl.store(lse_base + off_h_sub * stride_mh, lse, mask=h_mask_sub)

    out = acc / l_i[:, None]
    out_ptrs = out_base + off_h_sub[:, None] * stride_oh + off_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(out_base.dtype.element_ty), mask=h_mask_sub[:, None])


@triton.jit
def _attn_fwd(
    Q_ptr,
    KV_ptr,
    TopKIdx_ptr,
    Sink_ptr,
    LSE_ptr,
    Out_ptr,
    K_Buffer_ptr,
    QK_Buffer_ptr,
    P_Buffer_ptr,
    PV_Buffer_ptr,
    sm_scale,
    stride_qz,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kvz,
    stride_kvs,
    stride_kvd,
    stride_iz,
    stride_is,
    stride_in,
    stride_oz,
    stride_os,
    stride_oh,
    stride_od,
    stride_sink,
    stride_mz,
    stride_ms,
    stride_mh,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    """
    Top-level Triton kernel for Sparse Flash Attention.
    
    Responsibilities:
    1. Grid Mapping: Maps program IDs to [Batch, Sequence] dimensions.
    2. Pointer Arithmetic: Calculates base addresses for global tensors.
    3. Buffer Management: Offsets workspace pointers (K, V, QK, P, PV) per program.
    4. Head Iteration: Loops through Head dimension using BLOCK_H tiles.
    """
    
    # --- Parallel Mapping ---
    off_batch = tl.program_id(0).to(tl.int64)
    off_seq = tl.program_id(1).to(tl.int64)

    # --- Global Memory Indexing ---
    q_base = Q_ptr + off_batch * stride_qz + off_seq * stride_qs
    out_base = Out_ptr + off_batch * stride_oz + off_seq * stride_os
    idx_base = TopKIdx_ptr + off_batch * stride_iz + off_seq * stride_is
    lse_base = LSE_ptr + off_batch * stride_mz + off_seq * stride_ms
    kv_base = KV_ptr + off_batch * stride_kvz

    # --- Workspace (Buffer) Partitioning ---
    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)

    off_buf_kv = pid * (BLOCK_N * HEAD_DIM)
    off_buf_qk = pid * (BLOCK_H * BLOCK_N)
    off_buf_pv = pid * (BLOCK_H * HEAD_DIM)

    cur_k_buf = K_Buffer_ptr + off_buf_kv
    cur_qk_buf = QK_Buffer_ptr + off_buf_qk
    cur_p_buf = P_Buffer_ptr + off_buf_qk
    cur_pv_buf = PV_Buffer_ptr + off_buf_pv

    off_d = tl.arange(0, HEAD_DIM)

    # --- Head-Dimension Tiling ---
    for start_h in range(0, H, BLOCK_H):
        _inner_fwd(
            q_base,
            kv_base,
            idx_base,
            Sink_ptr,
            lse_base,
            out_base,
            cur_k_buf,
            cur_qk_buf,
            cur_p_buf,
            cur_pv_buf,
            stride_qh,
            stride_qd,
            stride_kvs,
            stride_kvd,
            stride_in, 
            stride_sink,
            stride_mh,
            stride_oh,
            stride_od,
            off_d,
            sm_scale, 
            HEAD_DIM,
            BLOCK_N,
            TOPK,
            START_H=start_h,
            BLOCK_H=BLOCK_H,
            H=H,
            KV_CTX=KV_CTX,
        )


@triton.jit
def _get_delta_split(
    Out_ptr,
    Grad_O_ptr,
    stride_oh,
    stride_od,
    stride_goh,
    stride_god,
    off_d,
    row_indices,
    h_mask_sub,
    HEAD_DIM: tl.constexpr
):
    out_ptrs = Out_ptr + row_indices[:, None] * stride_oh + off_d[None, :] * stride_od
    do_ptrs = Grad_O_ptr + row_indices[:, None] * stride_goh + off_d[None, :] * stride_god

    out = tl.load(out_ptrs, mask=h_mask_sub[:, None], other=0.0).to(tl.float32)
    grad_o = tl.load(do_ptrs, mask=h_mask_sub[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(out * grad_o, axis=1)
    return delta


@triton.jit
def _inner_dq(
    q_full,
    do_full,
    lse_sub,
    delta,
    KV_ptr,
    TopKIdx_ptr,
    k_buf_ptr,
    s_buf_ptr,
    dp_buf_ptr,
    ds_buf_ptr,
    dq_buf_ptr,
    stride_kvn,
    stride_kvd,
    stride_tk,
    off_d,
    sm_scale,
    sub_id,
    row_indices,
    n_offset_local,
    n_offset_local1,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    HALF_H: tl.constexpr = BLOCK_H // 2
    acc_dq = tl.zeros([HALF_H, HEAD_DIM], dtype=tl.float32)

    HALF_K: tl.constexpr = BLOCK_K // 2
    off_k_buf = tl.arange(0, BLOCK_K)
    buf_range_h = tl.arange(0, BLOCK_H)

    for i in range(NUM_BLOCKS):
        start_k = i * BLOCK_K
        start_k1 = i.to(tl.float32) * BLOCK_K

        # --- Stage 1: Load K ---
        off_k_local0 = start_k + n_offset_local + tl.arange(0, HALF_K)
        off_k_local1 = start_k1 + n_offset_local1 + tl.arange(0, HALF_K).to(tl.float32)
        k_mask = off_k_local1 < TOPK

        idx = tl.load(TopKIdx_ptr + off_k_local0 * stride_tk, mask=k_mask, other=-1).to(tl.float32)
        valid = k_mask & (idx >= 0)

        # address conflict
        dummy_idx = (KV_CTX - 1) - tl.arange(0, HALF_K)

        k_idx_optimized = tl.where(valid, idx, dummy_idx)
        k_ptrs = KV_ptr + k_idx_optimized[:, None].to(tl.int64) * stride_kvn + off_d[None, :] * stride_kvd
        k_val = tl.load(k_ptrs, mask=valid[:, None], other=0.0)

        buf_k_idx = n_offset_local + tl.arange(0, HALF_K)
        tl.store(k_buf_ptr + buf_k_idx[:, None] * HEAD_DIM + off_d[None, :], k_val)

        al.sync_block_set("vector", "cube", 0)

        # --- Stage 2: S, dP ---
        al.sync_block_wait("vector", "cube", 0)

        k_load = tl.load(k_buf_ptr + off_k_buf[:, None] * HEAD_DIM + off_d[None, :])

        s_res = tl.dot(q_full, tl.trans(k_load))
        dp_res = tl.dot(do_full, tl.trans(k_load)) # Reuse K

        tl.store(s_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :], s_res)
        tl.store(dp_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :], dp_res)

        # # --- Stage 3: P, dS ---  
        s_sub = tl.load(s_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :])
        dp_sub = tl.load(dp_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :])

        s_sub = s_sub * (sm_scale * LOG2_E)
        p_sub = tl.math.exp2(s_sub - lse_sub[:, None])

        off_k_global = start_k + off_k_buf
        idx_global_mask = off_k_global.to(tl.float32) < TOPK
        idx_global = tl.load(TopKIdx_ptr + off_k_global * stride_tk, mask=idx_global_mask, other=-1).to(tl.float32)
        valid_compute = (off_k_global.to(tl.float32) < TOPK) & (idx_global >= 0)

        p_sub = tl.where(valid_compute[None, :], p_sub, 0.0)
        ds_sub = p_sub * (dp_sub - delta[:, None])
        ds_sub = tl.where(valid_compute[None, :], ds_sub, 0.0)

        tl.store(ds_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :], ds_sub.to(tl.bfloat16))

        al.sync_block_set("vector", "cube", 2)

        # --- Stage 4: dQ_part ---
        al.sync_block_wait("vector", "cube", 2)

        ds_full = tl.load(ds_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :])
        dq_part = tl.dot(ds_full, k_load)

        tl.store(dq_buf_ptr + buf_range_h[:, None] * HEAD_DIM + off_d[None, :], dq_part)

        # # --- Stage 5: Accumulate dQ ---     
        dq_load = tl.load(dq_buf_ptr + row_indices[:, None] * HEAD_DIM + off_d[None, :])
        acc_dq += dq_load * sm_scale

    return acc_dq


@triton.jit
def _inner_dkv(
    q_full,
    do_full,
    lse_sub,
    delta,
    KV_ptr,
    TopKIdx_ptr,
    Grad_KV_ptr,
    k_buf_ptr,
    s_buf_ptr,
    dp_buf_ptr,
    p_buf_ptr,
    ds_buf_ptr,
    dk_buf_ptr,
    dv_buf_ptr,
    stride_kvn,
    stride_kvd,
    stride_tk,
    stride_gkvs,
    stride_gkvd,
    off_d,
    sm_scale,
    sub_id,
    row_indices, 
    n_offset_local,
    n_offset_local1,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    HALF_K: tl.constexpr = BLOCK_K // 2
    off_k_buf = tl.arange(0, BLOCK_K)
    buf_range_h = tl.arange(0, BLOCK_H)

    for i in range(NUM_BLOCKS):
        start_k = i * BLOCK_K
        start_k1 = i.to(tl.float32) * BLOCK_K

        # --- Stage 1: Load K ---
        off_k_local0 = start_k + n_offset_local + tl.arange(0, HALF_K)
        off_k_local1 = start_k1 + n_offset_local1 + tl.arange(0, HALF_K).to(tl.float32)
        k_mask = off_k_local1 < TOPK
        idx = tl.load(TopKIdx_ptr + off_k_local0 * stride_tk, mask=k_mask, other=-1).to(tl.int32)
        valid = k_mask & (idx.to(tl.float32) >= 0)

        # address conflict
        dummy_idx = (KV_CTX - 1) - tl.arange(0, HALF_K)

        k_idx_optimized = tl.where(valid, idx, dummy_idx)
        k_ptrs = KV_ptr + k_idx_optimized[:, None].to(tl.int64) * stride_kvn + off_d[None, :] * stride_kvd

        k_val = tl.load(k_ptrs, mask=valid[:, None], other=0.0)

        buf_k_idx = n_offset_local + tl.arange(0, HALF_K)
        tl.store(k_buf_ptr + buf_k_idx[:, None] * HEAD_DIM + off_d[None, :], k_val.to(tl.float32))
        al.sync_block_set("vector", "cube", 0)

        # --- Stage 2: S, dP ---
        al.sync_block_wait("vector", "cube", 0)
        k_load = tl.load(k_buf_ptr + off_k_buf[:, None] * HEAD_DIM + off_d[None, :])
        k_load_bf16 = k_load.to(tl.bfloat16)
        s_res = tl.dot(q_full, tl.trans(k_load_bf16))
        dp_res = tl.dot(do_full, tl.trans(k_load_bf16))
        tl.store(s_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :], s_res)
        tl.store(dp_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :], dp_res)
        al.sync_block_set("cube", "vector", 1)

        # --- Stage 3: P, dS ---
        al.sync_block_wait("cube", "vector", 1)
        s_sub = tl.load(s_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :]).to(tl.float32)
        dp_sub = tl.load(dp_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :]).to(tl.float32)
        s_sub = s_sub * (sm_scale * LOG2_E)
        p_sub = tl.math.exp2(s_sub - lse_sub[:, None])

        off_k_global0 = start_k + off_k_buf
        off_k_global1 = start_k.to(tl.float32) + off_k_buf.to(tl.float32)
        idx_global_mask = off_k_global1 < TOPK
        idx_global = tl.load(TopKIdx_ptr + off_k_global0 * stride_tk, mask=idx_global_mask, other=-1).to(tl.float32)
        valid_compute = (off_k_global1 < TOPK) & (idx_global >= 0)

        p_sub = tl.where(valid_compute[None, :], p_sub, 0.0)
        ds_sub = p_sub * (dp_sub - delta[:, None])
        ds_sub = tl.where(valid_compute[None, :], ds_sub, 0.0)

        tl.store(p_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :], p_sub.to(tl.bfloat16))
        tl.store(ds_buf_ptr + row_indices[:, None] * BLOCK_K + off_k_buf[None, :], ds_sub.to(tl.bfloat16))
        al.sync_block_set("vector", "cube", 2)

        # --- Stage 4 & 5: Serial Compute ---
        al.sync_block_wait("vector", "cube", 2)

        # Prepare shared params for atomic_add
        buf_k_idx0 = n_offset_local + tl.arange(0, HALF_K)
        buf_k_idx1 = n_offset_local.to(tl.float32) + tl.arange(0, HALF_K).to(tl.float32)
        idx_step_mask = (start_k.to(tl.float32) + buf_k_idx1) < TOPK
        idx_step = tl.load(TopKIdx_ptr + (start_k + buf_k_idx0) * stride_tk, mask=idx_step_mask, other=-1).to(tl.float32)
        valid_step = idx_step_mask & (idx_step >= 0)

        # address conflict
        dummy_idx_step = (KV_CTX - 1) - tl.arange(0, HALF_K)

        idx_safe_step = tl.where(valid_step, idx_step, dummy_idx_step)
        gk_ptrs = Grad_KV_ptr + idx_safe_step[:, None].to(tl.int64) * stride_gkvs + off_d[None, :] * stride_gkvd

        # -----------------------------------------------------------
        # [Part A] Compute dV
        # -----------------------------------------------------------
        p_full = tl.load(p_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :])
        dv_part = tl.dot(tl.trans(p_full.to(tl.bfloat16)), do_full)

        tl.store(dv_buf_ptr + off_k_buf[:, None] * HEAD_DIM + off_d[None, :], dv_part)

        dv_val = tl.load(dv_buf_ptr + buf_k_idx0[:, None] * HEAD_DIM + off_d[None, :]).to(tl.float32)

        # -----------------------------------------------------------
        # [Part B] Compute dK
        # -----------------------------------------------------------
        ds_full = tl.load(ds_buf_ptr + buf_range_h[:, None] * BLOCK_K + off_k_buf[None, :])
        dk_part = tl.dot(tl.trans(ds_full.to(tl.bfloat16)), q_full)

        tl.store(dk_buf_ptr + off_k_buf[:, None] * HEAD_DIM + off_d[None, :], dk_part)

        dk_val = tl.load(dk_buf_ptr + buf_k_idx0[:, None] * HEAD_DIM + off_d[None, :]).to(tl.float32)
        dk_grad = dk_val * sm_scale

        # -----------------------------------------------------------
        # [Merged] Single atomic_add for dV + dK (halves atomic ops)
        # -----------------------------------------------------------
        combined_grad = dv_val + dk_grad
        combined_grad = tl.where(valid_step[:, None], combined_grad, 0.0)
        tl.atomic_add(gk_ptrs, combined_grad, mask=valid_step[:, None])


@triton.jit
def _attn_bwd_dq_dsink(
    Q_ptr,
    KV_ptr,
    Sink_ptr,
    TopKIdx_ptr,
    grad_out_ptr,
    grad_q_ptr,
    grad_sink_ptr,
    LSE_ptr,
    Out_ptr,
    k_buf_ptr,
    s_buf_ptr,
    dp_buf_ptr,
    ds_buf_ptr,
    dq_buf_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_gob,
    stride_gom,
    stride_goh,
    stride_god,
    stride_gqb,
    stride_gqm,
    stride_gqh,
    stride_gqd,
    stride_tb,
    stride_tm,
    stride_tk,
    stride_lseb,
    stride_lsem,
    stride_lseh,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_sink,
    stride_gsink,
    sm_scale,
    TOPK: tl.constexpr,
    n_ctx: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    off_batch = tl.program_id(axis=0)
    off_seq = tl.program_id(axis=1)

    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    off_buf_kv = pid * (BLOCK_K * head_dim)
    off_buf_qk = pid * (BLOCK_H * BLOCK_K)
    off_buf_dq = pid * (BLOCK_H * head_dim)

    cur_k_buf = k_buf_ptr + off_buf_kv
    cur_s_buf = s_buf_ptr + off_buf_qk
    cur_dp_buf = dp_buf_ptr + off_buf_qk
    cur_ds_buf = ds_buf_ptr + off_buf_qk
    cur_dq_buf = dq_buf_ptr + off_buf_dq

    q_base = Q_ptr + off_batch * stride_qb + off_seq * stride_qm
    grad_o_base = grad_out_ptr + off_batch * stride_gob + off_seq * stride_gom
    lse_base = LSE_ptr + off_batch * stride_lseb + off_seq * stride_lsem
    out_base = Out_ptr + off_batch * stride_ob + off_seq * stride_om
    grad_q_base = grad_q_ptr + off_batch * stride_gqb + off_seq * stride_gqm
    topk_ptr_base = TopKIdx_ptr + off_batch * stride_tb + off_seq * stride_tm
    kv_ptr_base = KV_ptr + off_batch * stride_kvb 

    off_d = tl.arange(0, head_dim)

    for start_h in range(0, n_heads, BLOCK_H):
        off_h_full = start_h + tl.arange(0, BLOCK_H)
        h_mask_full = off_h_full < n_heads

        sub_id = al.sub_vec_id()
        HALF_H: tl.constexpr = BLOCK_H // 2
        row_indices = tl.arange(0, HALF_H) + sub_id * HALF_H
        off_h_sub = start_h + row_indices
        h_mask_sub = off_h_sub < n_heads
        HALF_K: tl.constexpr = BLOCK_K // 2
        n_offset_local = sub_id * HALF_K
        n_offset_local1 = sub_id.to(tl.float32) * HALF_K

        q_ptrs = q_base + off_h_full[:, None] * stride_qh + off_d[None, :] * stride_qd
        do_ptrs = grad_o_base + off_h_full[:, None] * stride_goh + off_d[None, :] * stride_god
        q_full = tl.load(q_ptrs, mask=h_mask_full[:, None], other=0.0)
        do_full = tl.load(do_ptrs, mask=h_mask_full[:, None], other=0.0)

        delta = _get_delta_split(
            out_base,
            grad_o_base,
            stride_oh,
            stride_od,
            stride_goh,
            stride_god,
            off_d,
            off_h_sub,
            h_mask_sub,
            head_dim
        )

        sink_ptr_now = Sink_ptr + off_h_sub
        gsink_ptr_now = grad_sink_ptr + off_h_sub
        sink_val = tl.load(sink_ptr_now, mask=h_mask_sub, other=0.0)
        lse_val = tl.load(lse_base + off_h_sub * stride_lseh, mask=h_mask_sub, other=0.0)
        p_sink = tl.math.exp2(sink_val * LOG2_E - lse_val)
        d_sink = -p_sink * delta
        tl.atomic_add(gsink_ptr_now, d_sink, mask=h_mask_sub)

        acc_dq = _inner_dq(
            q_full,
            do_full,
            lse_val,
            delta,
            kv_ptr_base,
            topk_ptr_base,
            cur_k_buf,
            cur_s_buf,
            cur_dp_buf,
            cur_ds_buf,
            cur_dq_buf,
            stride_kvn,
            stride_kvd,
            stride_tk,
            off_d,
            sm_scale,
            sub_id,
            row_indices,
            n_offset_local,
            n_offset_local1,
            head_dim,
            BLOCK_K,
            BLOCK_H,
            TOPK,
            NUM_BLOCKS,
            KV_CTX,
        )

        gq_ptrs = grad_q_base + off_h_sub[:, None] * stride_gqh + off_d[None, :] * stride_gqd
        tl.store(gq_ptrs, acc_dq.to(tl.bfloat16), mask=h_mask_sub[:, None])


@triton.jit
def _attn_bwd_dk_dv(
    Q_ptr,
    KV_ptr,
    TopKIdx_ptr,
    grad_out_ptr,
    grad_kv_ptr,
    LSE_ptr,
    Out_ptr,
    k_buf_ptr,
    s_buf_ptr,
    dp_buf_ptr,
    p_buf_ptr,
    ds_buf_ptr,
    dk_buf_ptr,
    dv_buf_ptr,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_gob,
    stride_gom,
    stride_goh,
    stride_god,
    stride_gkvs,
    stride_gkvd,
    stride_tb,
    stride_tm,
    stride_tk,
    stride_lseb,
    stride_lsem,
    stride_lseh,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    sm_scale,
    TOPK: tl.constexpr,
    n_ctx: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_CTX: tl.constexpr,
):
    off_batch = tl.program_id(axis=0)
    off_seq = tl.program_id(axis=1)

    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    off_buf_kv = pid * (BLOCK_K * head_dim)
    off_buf_qk = pid * (BLOCK_H * BLOCK_K)
    off_buf_dk = pid * (BLOCK_K * head_dim)

    cur_k_buf = k_buf_ptr + off_buf_kv
    cur_s_buf = s_buf_ptr + off_buf_qk
    cur_dp_buf = dp_buf_ptr + off_buf_qk
    cur_p_buf = p_buf_ptr + off_buf_qk
    cur_ds_buf = ds_buf_ptr + off_buf_qk
    cur_dk_buf = dk_buf_ptr + off_buf_dk
    cur_dv_buf = dv_buf_ptr + off_buf_dk

    q_base = Q_ptr + off_batch * stride_qb + off_seq * stride_qm
    grad_o_base = grad_out_ptr + off_batch * stride_gob + off_seq * stride_gom
    lse_base = LSE_ptr + off_batch * stride_lseb + off_seq * stride_lsem
    out_base = Out_ptr + off_batch * stride_ob + off_seq * stride_om
    topk_ptr_base = TopKIdx_ptr + off_batch * stride_tb + off_seq * stride_tm
    kv_ptr_base = KV_ptr + off_batch * stride_kvb 
    grad_kv_ptr_base = grad_kv_ptr + off_batch * stride_kvb

    off_d = tl.arange(0, head_dim)

    for start_h in range(0, n_heads, BLOCK_H):
        off_h_full = start_h + tl.arange(0, BLOCK_H)
        h_mask_full = off_h_full < n_heads

        sub_id = al.sub_vec_id()
        HALF_H: tl.constexpr = BLOCK_H // 2
        row_indices = tl.arange(0, HALF_H) + sub_id * HALF_H
        off_h_sub = start_h + row_indices
        h_mask_sub = off_h_sub < n_heads
        HALF_K: tl.constexpr = BLOCK_K // 2
        n_offset_local = sub_id * HALF_K
        n_offset_local1 = sub_id.to(tl.float32) * HALF_K

        q_ptrs = q_base + off_h_full[:, None] * stride_qh + off_d[None, :] * stride_qd
        do_ptrs = grad_o_base + off_h_full[:, None] * stride_goh + off_d[None, :] * stride_god
        q_full = tl.load(q_ptrs, mask=h_mask_full[:, None], other=0.0)
        do_full = tl.load(do_ptrs, mask=h_mask_full[:, None], other=0.0)

        delta = _get_delta_split(
            out_base,
            grad_o_base,
            stride_oh,
            stride_od,
            stride_goh,
            stride_god,
            off_d,
            off_h_sub,
            h_mask_sub,
            head_dim
        )

        lse_val = tl.load(lse_base + off_h_sub * stride_lseh, mask=h_mask_sub, other=0.0)

        _inner_dkv(
            q_full,
            do_full,
            lse_val,
            delta,
            kv_ptr_base,
            topk_ptr_base,
            grad_kv_ptr_base,
            cur_k_buf,
            cur_s_buf,
            cur_dp_buf,
            cur_p_buf,
            cur_ds_buf,
            cur_dk_buf,
            cur_dv_buf,
            stride_kvn,
            stride_kvd,
            stride_tk,
            stride_gkvs,
            stride_gkvd,
            off_d,
            sm_scale,
            sub_id,
            row_indices,
            n_offset_local,
            n_offset_local1,
            head_dim,
            BLOCK_K,
            BLOCK_H,
            TOPK,
            NUM_BLOCKS,
            KV_CTX,
        )
