"""Run real FSDP updates for speed/memory validation; never save weights or start eval/W&B.

torchrun --nproc_per_node=2 ops/h20-benchmark.py --micro-batch 2 --backend _flash_3_varlen_hub
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.distributed as dist

from rdm.train.launch import load_config
from rdm.utils.distributed import setup_distributed
from rdm.utils.logging import setup_logging
from rdm.utils.seed import fix_random_seeds
from zimage_distill.launch import build_trainer

ap = argparse.ArgumentParser()
ap.add_argument('--micro-batch', type=int, choices=[1, 2, 4, 8, 16], required=True)
ap.add_argument('--backend', default='native', choices=['native', '_flash_3_varlen_hub'])
ap.add_argument('--iterations', type=int, default=3)
a = ap.parse_args()
rank, world, local_rank = setup_distributed()
assert world == 2 and 32 % (world * a.micro_batch) == 0
setup_logging(rank=rank)
fix_random_seeds(0)
cfg = load_config('z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml')
cfg.reference_root = '/data/shaoshitong/rdm-sets/zimage/zimage-ref-v1/reference_store_qwen'
cfg.memory['micro_batch'] = a.micro_batch
cfg.zimage['attention_backend'] = a.backend
# Benchmark the steady-state generator adversarial path without burning 50 formal steps.
cfg.gan['g_start_step'] = 0
device = f'cuda:{local_rank}'
trainer = build_trainer(cfg, device=device)
trainer.bootstrap()
dist.barrier()
rows = []
for i in range(a.iterations):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    logs = trainer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert logs.get('skipped') is False, logs
    bad = [k for k, v in logs.items() if isinstance(v, (int, float)) and not math.isfinite(v)]
    assert not bad, bad
    row = {'iteration': i + 1, 'seconds': elapsed,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'logs': logs}
    gathered = [None] * world
    dist.all_gather_object(gathered, row)
    if rank == 0:
        rows.append(gathered)
        print(json.dumps({'micro_batch': a.micro_batch, 'grad_accum': 32 // (world * a.micro_batch),
                          'backend': a.backend, 'ranks': gathered}), flush=True)
if rank == 0:
    out = Path('/data/shaoshitong/rdm-sets/control') / f'benchmark-{a.backend}-micro{a.micro_batch}.json'
    out.write_text(json.dumps({'micro_batch': a.micro_batch, 'backend': a.backend,
                               'global_batch': 32, 'generator_gan_active': True, 'steps': rows}, indent=2))
dist.destroy_process_group()
