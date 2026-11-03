"""Assemble only the tables already downloaded and hash-verified at pinned revisions."""
import json
import os
from pathlib import Path
import numpy as np
from zimage_distill.qwen_text_store import BUILT, link_source, rows_read, stats_rows

root = Path('/data/shaoshitong/rdm-sets')
pack = root / 'zimage/zimage-ref-v1'
source = pack / 'reference_store'
download = root / 'zimage/qwen-text-download'
out = pack / 'reference_store_qwen'
base = json.loads((source / 'metadata.json').read_text())
meta = json.loads((download / 'metadata.json').read_text())
assert base['prompt_hash'] == meta['prompt_hash']
assert base['num_rows'] == meta['num_rows']
assert meta['text_dim'] == 2560
tau = np.load(download / 'text_features.npy', mmap_mode='r')
assert tau.shape == (base['num_prompts'], 2560)
order = np.load(source / 'row_order.npy', mmap_mode='r')
ids = np.unique(np.concatenate([rows_read(order, 500, 128, 32),
                               stats_rows(base['num_rows'], 8192, [0, 1])]))
prompt_ids = np.load(source / 'prompt_ids.npy', mmap_mode='r')[ids]
checked = tau[np.unique(prompt_ids)]
assert np.isfinite(checked).all()
assert np.allclose(np.linalg.norm(checked, axis=1), 1, atol=1e-3)
if not out.exists():
    tmp = out.with_name(out.name + '.partial')
    tmp.mkdir(exist_ok=True)
    for name in BUILT:
        dest = tmp / name
        if not dest.exists():
            os.link(download / name, dest)
    link_source(source, tmp)
    tmp.rename(out)
receipt = {'store': str(out), 'text_dim': 2560, 'required_rows': len(ids),
           'required_prompts': len(checked), 'steps': 500, 'window': 128,
           'global_batch': 32, 'critic_seeds': [0, 1], 'finite_and_unit_norm': True,
           'metadata': meta}
(root / 'control/qwen-store-assembled.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
