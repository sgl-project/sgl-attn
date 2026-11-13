"""Long-KV guards for the paged SM90 FP8 two-level accumulator."""

import math
import os

import pytest
import torch

from flash_attn_interface import flash_attn_with_kvcache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9
    or os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE",
    reason="requires the SM90 FP8 forward kernel",
)


def paged_inputs(seqlen_q, seqlen_k, padded_q=0):
    torch.manual_seed(20261110)
    q = torch.randn(seqlen_q + padded_q, 12, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(math.ceil(seqlen_k / 8), 8, 2, 256, device="cuda", dtype=torch.bfloat16)
    return (
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        torch.randperm(k.shape[0], device="cuda", dtype=torch.int32)[None],
    )


def attention_operation(q, k, v, page_table, seqlen_q, seqlen_k, num_splits):
    cache_seqlens = torch.tensor([seqlen_k], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, seqlen_q], device="cuda", dtype=torch.int32)
    cu_seqlens_k_new = torch.tensor([0, seqlen_k], device="cuda", dtype=torch.int32)

    def operation():
        return flash_attn_with_kvcache(
            q=q,
            k_cache=k,
            v_cache=v,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            max_seqlen_q=seqlen_q,
            softmax_scale=256**-0.5,
            causal=True,
            num_splits=num_splits,
        )

    return operation


@pytest.mark.parametrize("seqlen_q", [7, 53, 16384])
@pytest.mark.parametrize("seqlen_k", [16384, 65536, 262144, 520000])
@pytest.mark.parametrize("num_splits", [0, 1])
@torch.inference_mode()
def test_fp8_paged_long_kv_constant_values(seqlen_q, seqlen_k, num_splits):
    q, k, page_table = paged_inputs(seqlen_q, seqlen_k)
    v = torch.ones_like(k)
    result = attention_operation(q, k, v, page_table, seqlen_q, seqlen_k, num_splits)()
    assert torch.isfinite(result).all()
    # V=1 gives an analytic reference without materializing a long attention
    # matrix. FP8 probability rounding can leave a small residual, but the old
    # reduced-precision running accumulator develops a much larger long-KV bias.
    rms_error = (result.float() - 1).square().mean().sqrt()
    assert rms_error.item() < 0.02


@pytest.mark.parametrize("num_splits", [0, 1])
@pytest.mark.parametrize("cuda_graph", [False, True])
@torch.inference_mode()
def test_fp8_paged_tail_and_padded_q(num_splits, cuda_graph):
    seqlen_q, seqlen_k = 7, 65539
    q, k, page_table = paged_inputs(seqlen_q, seqlen_k, padded_q=13)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16).to(k.dtype)
    saved_k, saved_v = k.clone(), v.clone()

    operation = attention_operation(q, k, v, page_table, seqlen_q, seqlen_k, num_splits)

    if cuda_graph:
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = operation()
        graph.replay()
    else:
        result = operation()
    torch.cuda.synchronize()
    assert torch.equal(k.view(torch.uint8), saved_k.view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), saved_v.view(torch.uint8))
    result = result[:seqlen_q].double()
    assert torch.isfinite(result).all()

    # Check the same FP8-rounded Q/K/V against a compact grouped FP64 reference.
    logical_k = k[page_table[0].long()].reshape(-1, 2, 256)[:seqlen_k].double()
    logical_v = v[page_table[0].long()].reshape(-1, 2, 256)[:seqlen_k].double()
    grouped_q = q[:seqlen_q].double().reshape(seqlen_q, 2, 6, 256)
    grouped_q = grouped_q.permute(1, 2, 0, 3).reshape(2, 6 * seqlen_q, 256)
    scores = (grouped_q @ logical_k.permute(1, 2, 0) * 256**-0.5).view(2, 6, seqlen_q, seqlen_k)
    mask = torch.arange(seqlen_k, device="cuda")[None, :] > (
        torch.arange(seqlen_q, device="cuda")[:, None] + seqlen_k - seqlen_q
    )
    scores.masked_fill_(mask[None, None], -torch.inf)
    reference = scores.softmax(-1).reshape(2, 6 * seqlen_q, seqlen_k) @ logical_v.permute(1, 0, 2)
    reference = reference.view(2, 6, seqlen_q, 256).permute(2, 0, 1, 3).reshape(result.shape)
    relative_error = (result - reference).norm() / reference.norm().clamp_min(1e-30)
    assert relative_error.item() < 0.08
