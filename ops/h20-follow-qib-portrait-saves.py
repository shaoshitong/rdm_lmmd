#!/usr/bin/env python3
"""Retain the user-requested identical portraits at future existing saves.

Never changes, starts, or restarts training; preserves PNGs, never checkpoint copies.
"""
import datetime, json, os, pathlib, subprocess, time
import torch

ROOT = pathlib.Path('/data/shaoshitong/rdm-sets')
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
OUT = ROOT/'visualizations/twohead-lr10-A/qib-portraits-v1'
REPO = pathlib.Path(__file__).resolve().parents[1]
PREFIX = os.environ.get('H20_CONTROL_PREFIX','twohead-lr10-A-500')
FOLLOW_PREFIX = os.environ.get('H20_PORTRAIT_FOLLOW_PREFIX','twohead-qib-portraits-follow')


def write(value):
    p = ROOT/'control'/f'{FOLLOW_PREFIX}-progress.json'
    value['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = p.with_suffix('.partial'); tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, p)


def main():
    torch.set_num_threads(1)
    deadline = time.monotonic()+24*3600
    pending = list(range(200, 501, 50))
    results = []
    seen = None
    actual = None
    while pending and time.monotonic() < deadline:
        initial = OUT/'step_0000000/preview-verified.json'
        first_exit = ROOT/'control/twohead-qib-portraits-150.exit'
        if first_exit.exists() and first_exit.read_text().strip() != '0':
            raise RuntimeError('Initial0/150 preview failed; inspect retained command and log')
        ckpt = EXPERIMENT/'resume.pth'
        if ckpt.exists():
            st = ckpt.stat(); stamp = (st.st_ino, st.st_size, st.st_mtime_ns)
            if stamp != seen:
                state = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
                actual = int(state['step']); del state
                seen = stamp
        target = pending[0]
        if actual is not None and actual > target:
            results.append(dict(step=target, status='missed', reason='Rolling checkpoint already newer'))
            pending.pop(0)
        elif actual == target and initial.exists() and first_exit.exists() and first_exit.read_text().strip() == '0':
            proof_path = OUT/f'step_{target:07d}'/'preview-verified.json'
            exit_path = ROOT/f'control/twohead-qib-portraits-{target}.exit'
            if proof_path.exists() and exit_path.exists() and exit_path.read_text().strip() == '0':
                assert json.loads(proof_path.read_text())['passed']
                results.append(dict(step=target, status='verified_existing')); pending.pop(0)
            else:
                used, total = map(int, subprocess.check_output(['nvidia-smi','-i','6',
                    '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'], text=True).strip().split(','))
                if total-used >= 48*1024:
                    command = ['bash', str(REPO/'ops/h20-run-qib-portrait-preview.sh'), str(target), '6']
                    print(json.dumps(dict(event='preview_start', step=target, command=command)), flush=True)
                    proc = subprocess.run(command, timeout=1860)
                    if proc.returncode != 0:
                        write(dict(pending=pending, results=results, failed_step=target, exit=proc.returncode))
                        raise RuntimeError(f'Stage{target} portrait failed with true exit{proc.returncode}')
                    proof = json.loads(proof_path.read_text())
                    assert proof['passed'] and len(proof['files']) == 16 and len(proof['native_comparisons']) == 4
                    results.append(dict(step=target, status='verified', images=16)); pending.pop(0)
        write(dict(checkpoint_step=actual, pending=pending, results=results))
        training_exit = ROOT/'control'/f'{PREFIX}.exit'
        if pending and training_exit.exists() and training_exit.read_text().strip() != '0':
            raise RuntimeError('Training exited nonzero; portrait follow-up stops without restart')
        if pending:
            time.sleep(20)
    assert not pending, 'Portrait follow-up exceeded24h deadline'
    assert all(r['status'] != 'missed' for r in results), 'Some requested stages could not be retained'
    write(dict(passed=True, pending=[], results=results))


if __name__ == '__main__':
    main()
