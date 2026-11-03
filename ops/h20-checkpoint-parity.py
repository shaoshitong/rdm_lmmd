"""Compare no-checkpoint / per-block / two-block checkpointing under real two-rank FSDP.

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=z-image:. torchrun --nproc_per_node=2 \
    --master_port=29541 ops/h20-checkpoint-parity.py --out <remote-control>/checkpoint-parity.json

Only a tiny fp32 model is constructed. No model weights, optimizer, formal run, or evaluation
are saved. Each configuration is destroyed before constructing the next one.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from zimage_distill.checkpointing import configure_checkpoint_groups


class TinyBlock(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(0.17)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.down(self.dropout(torch.nn.functional.gelu(self.up(x))))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_refiner = nn.ModuleList()
        self.context_refiner = nn.ModuleList()
        self.layers = nn.ModuleList(TinyBlock() for _ in range(5))
        self.projection = nn.Linear(32, 16)
        self.gradient_checkpointing = True

    def forward(self, x, controlnet_block_samples=None):
        for layer in self.layers:
            x = checkpoint(layer.__call__, x, use_reentrant=False) \
                if torch.is_grad_enabled() and self.gradient_checkpointing else layer(x)
        return self.projection(x)


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.double(), expected.double()
    return {
        'max_absolute': float((a - b).abs().max()) if a.numel() else 0.0,
        'relative_l2': float((a - b).norm() / b.norm().clamp_min(1e-20)),
    }


def assert_same(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    return difference(actual, expected)


def run_mode(mode: str, state: dict, keys: list[str], local_rank: int, rank: int) -> dict:
    toy = TinyTransformer()
    toy.load_state_dict(state, strict=True)
    toy.gradient_checkpointing = mode != 'none'
    if mode != 'none':
        configure_checkpoint_groups(toy, 1 if mode == 'block1' else 2)
    assert sorted(toy.state_dict()) == keys, 'Checkpoint grouping changed canonical state keys'
    original_blocks = tuple(toy.layers._modules.values())
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TinyBlock})
    model = FSDP(
        toy, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True, device_id=torch.device('cuda', local_rank),
    )
    assert all(isinstance(block, FSDP) for block in toy.layers._modules.values()), \
        'Each actual TinyBlock must remain its own FSDP unit'
    # Different inputs on each rank exercise real gradient averaging. The same local input,
    # target and dropout RNG are recreated independently for each checkpoint configuration.
    g = torch.Generator().manual_seed(1201 + rank)
    x = torch.randn(2, 7, 32, generator=g).to(local_rank).requires_grad_()
    target = torch.randn(2, 7, 16, generator=g).to(local_rank)
    torch.manual_seed(902 + rank)
    torch.cuda.manual_seed(902 + rank)
    dist.barrier()
    output = model(x)
    loss = (output - target).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        gradients[name] = None if gradient is None else gradient.detach().cpu().clone()
    assert any(g is not None and g.numel() for g in gradients.values())
    calls = [block.calls for block in original_blocks]
    assert calls == ([1] * 5 if mode == 'none' else [2] * 5), (mode, calls)
    result = {
        'mode': mode, 'output': output.detach().cpu().clone(),
        'input_gradient': x.grad.detach().cpu().clone(), 'gradients': gradients,
        'block_calls': calls, 'loss': float(loss.detach()),
    }
    # Keep only CPU copies. Avoid holding multiple FSDP model instances at once.
    del loss, output, x, target, model, toy, original_blocks, parameter, gradient
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4,5', 'Only physical GPUs4/5 are authorized'
    assert int(os.environ['WORLD_SIZE']) == 2
    local_rank, rank = int(os.environ['LOCAL_RANK']), int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(71)
        template = TinyTransformer()
        state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
        keys = sorted(state)
        del template
        baseline = run_mode('none', state, keys, local_rank, rank)
        summary = {'rank': rank, 'baseline_loss': baseline['loss'], 'modes': {}}
        for mode in ('block1', 'block2'):
            actual = run_mode(mode, state, keys, local_rank, rank)
            assert actual['gradients'].keys() == baseline['gradients'].keys()
            parameter_differences = []
            for name, gradient in actual['gradients'].items():
                reference = baseline['gradients'][name]
                assert (gradient is None) == (reference is None), name
                if gradient is not None:
                    parameter_differences.append(assert_same(gradient, reference))
            summary['modes'][mode] = {
                'output': assert_same(actual['output'], baseline['output']),
                'input_gradient': assert_same(actual['input_gradient'], baseline['input_gradient']),
                'parameter_gradient_max_absolute': max(
                    row['max_absolute'] for row in parameter_differences),
                'parameter_gradient_max_relative_l2': max(
                    row['relative_l2'] for row in parameter_differences),
                'local_parameter_tensors_with_grad': len(parameter_differences),
                'block_calls': actual['block_calls'], 'loss': actual['loss'], 'passed': True,
            }
            del actual
        ranks = [None] * 2
        dist.all_gather_object(ranks, summary)
        if rank == 0:
            record = {
                'passed': True, 'world_size': 2, 'dtype': 'float32',
                'torch': torch.__version__, 'dropout': 0.17, 'layers': 5,
                'canonical_state_keys': len(keys),
                'canonical_state_keys_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
                'fsdp_unit': 'individual TinyBlock', 'use_orig_params': True, 'ranks': ranks,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.out.with_name(args.out.name + f'.tmp-{os.getpid()}')
            tmp.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(tmp, args.out)
            print(json.dumps(record), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
