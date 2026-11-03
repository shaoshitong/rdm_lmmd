"""Bounded independent GPU4/5 capacity/throughput search; never launch formal A/B."""
import datetime as dt
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/data/shaoshitong/rdm-sets')
parser = argparse.ArgumentParser()
parser.add_argument('--resume-summary', type=Path)
parser.add_argument('--only-case', type=int, nargs=2, metavar=('GROUP', 'MICRO'))
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[1]
STAMP = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = ROOT / 'control/checkpoint-sweep' / STAMP
OUT.mkdir(parents=True)
LATEST = ROOT / 'control/checkpoint-sweep-latest.json'
state = {'started_utc': STAMP, 'source': str(REPO),
         'source_commit': (REPO / 'SOURCE_COMMIT').read_text().strip(),
         'output': str(OUT), 'global_batch': 32, 'physical_gpus': [4, 5],
         'formal_training_started': False, 'status': 'running', 'cases': []}


def terminate(signum, frame):
    # Convert TERM to an exception so an active test's finally cleanup runs.
    raise SystemExit(f'Controller received signal {signum}')


signal.signal(signal.SIGTERM, terminate)


def publish():
    state['checked_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    tmp = LATEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, LATEST)
    (OUT / 'summary.json').write_text(json.dumps(state, indent=2))


def gpu_state():
    text = subprocess.check_output(['nvidia-smi', '-i', '4,5',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [list(map(int, row.split(','))) for row in text.splitlines()]


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError('Nonfinite benchmark/probe/refresh result')


def stop_test_tree(process, case_dir):
    # Elastic ranks have separate sessions. Snapshot descendants rather than
    # assuming killpg(torchrun) reaches them; PID start ticks prevent PID reuse.
    owned = []
    def collect(pid):
        p = Path('/proc') / str(pid)
        try:
            stat = (p / 'stat').read_text().split(') ', 1)[1].split()
            owned.append((pid, int(stat[19])))
            for child in (p / 'task' / str(pid) / 'children').read_text().split():
                collect(int(child))
        except (OSError, ValueError):
            return
    if process.poll() is None:
        collect(process.pid)
    # An elastic parent can already have exited while a rank's independent
    # session remains. Match this exact task entry+case, never an arbitrary GPU PID.
    entry = str(REPO / 'ops/h20-benchmark.py')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = [os.fsdecode(x) for x in (p / 'cmdline').read_bytes().split(b'\0')[:-1]]
            if entry in args and '--out-dir' in args and args[args.index('--out-dir') + 1] == str(case_dir):
                collect(int(p.name))
        except (OSError, ValueError, IndexError):
            continue
    owned = list(dict.fromkeys(owned))
    def signal_same(pid, ticks, sig):
        try:
            stat = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()
            if int(stat[19]) == ticks:
                os.kill(pid, sig)
        except (OSError, ValueError):
            pass
    for pid, ticks in reversed(owned):
        signal_same(pid, ticks, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for pid, ticks in reversed(owned):
        signal_same(pid, ticks, signal.SIGKILL)
    process.wait()
    return [{'pid': pid, 'start_ticks': ticks} for pid, ticks in owned]


def run_case(group, micro, iterations=2, label=None):
    name = label or f'group{group}-micro{micro}'
    case_dir = OUT / name
    case_dir.mkdir()
    idle = gpu_state()
    if any(memory > 32 or utilization != 0 for _, memory, total, utilization in idle):
        raise RuntimeError(f'Selected GPU busy before {name}: {idle}')
    command = [str(ROOT / 'envs/rdm/bin/torchrun'), '--nproc_per_node=2', '--master_port=29541',
               str(REPO / 'ops/h20-benchmark.py'), '--backend', '_flash_3_varlen_hub',
               '--micro-batch', str(micro), '--checkpoint-group-size', str(group),
               '--iterations', str(iterations), '--probe-refresh', '--out-dir', str(case_dir)]
    row = {'name': name, 'group_size': group, 'micro_batch': micro,
           'grad_accum': 16 // micro, 'command': command, 'status': 'running',
           'log': str(case_dir / 'run.log'), 'nvml_peak_mib': {str(x[0]): x[1] for x in idle}}
    state['cases'].append(row)
    publish()
    started = time.monotonic()
    with (case_dir / 'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        row['pid'] = process.pid
        try:
            while process.poll() is None:
                for index, used, total, utilization in gpu_state():
                    row['nvml_peak_mib'][str(index)] = max(row['nvml_peak_mib'][str(index)], used)
                if time.monotonic() - started > 1200:
                    row['timeout'] = True
                    row['cleanup'] = stop_test_tree(process, case_dir)
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                row['cleanup'] = stop_test_tree(process, case_dir)
        row['exit'] = process.wait()
        if row['exit'] != 0 and 'cleanup' not in row:
            row['cleanup'] = stop_test_tree(process, case_dir)
    row['wall_seconds'] = time.monotonic() - started
    text = (case_dir / 'run.log').read_text(errors='replace')
    result = case_dir / f'benchmark-_flash_3_varlen_hub-micro{micro}-group{group}.json'
    if row['exit'] == 0:
        record = json.loads(result.read_text())
        require_finite(record)
        assert len(record['steps']) == iterations
        assert all(not rank['logs']['skipped'] for step in record['steps'] for rank in step)
        assert record['probe_refresh'] is not None
        row['status'] = 'passed'
        row['result'] = str(result)
        # First update initializes kernels/optimizer state; use later complete updates.
        row['mean_seconds'] = sum(max(x['seconds'] for x in step)
                                  for step in record['steps'][1:]) / (iterations - 1)
        all_rows = [rank for step in record['steps'] for rank in step] + record['probe_refresh']
        row['peak_allocated_bytes'] = max(x['peak_allocated_bytes'] for x in all_rows)
        row['peak_reserved_bytes'] = max(x['peak_reserved_bytes'] for x in all_rows)
    elif 'out of memory' in text.lower() and ('CUDA' in text or 'torch.OutOfMemoryError' in text):
        row['status'] = 'oom'
        row['last30'] = text.splitlines()[-30:]
    else:
        row['status'] = 'failed'
        row['last30'] = text.splitlines()[-30:]
    publish()
    # torchrun reaps failed peers; do not start a case until its CUDA contexts disappear.
    for _ in range(30):
        if all(used <= 32 and utilization == 0 for _, used, total, utilization in gpu_state()):
            break
        time.sleep(1)
    else:
        raise RuntimeError('GPU contexts remain after completed test; stopping search')
    if row['status'] == 'failed':
        raise RuntimeError(f'Unexpected benchmark failure: {name}; see {row["log"]}')
    return row


try:
    if args.resume_summary:
        previous = json.loads(args.resume_summary.read_text())
        for item in previous['cases']:
            if item['status'] == 'passed':
                record = json.loads(Path(item['result']).read_text())
                require_finite(record)
                assert len(record['steps']) >= 2 and record['probe_refresh'] is not None
                assert all(not rank['logs']['skipped'] for step in record['steps'] for rank in step)
                reused = dict(item, reused_from=str(args.resume_summary))
                state['cases'].append(reused)
    publish()
    # Requested pair checkpointing first. Every successful point also exercises probe/refresh.
    plan = [(2, [2, 4, 8, 16]), (1, [4, 8, 16]), (0, [1, 2, 4, 8, 16])]
    if args.only_case:
        group, micro = args.only_case
        assert group in (0, 1, 2) and micro in (1, 2, 4, 8, 16)
        plan = [(group, [micro])]
    for group, batches in plan:
        for micro in batches:
            if any(x['group_size'] == group and x['micro_batch'] == micro and
                   x['status'] == 'passed' for x in state['cases']):
                continue
            point = run_case(group, micro, iterations=3 if args.only_case or (group, micro) == (2, 2) else 2)
            if point['status'] == 'oom':
                break
    if args.only_case:
        state['status'] = 'single_case_complete'
        state['priority_case'] = args.only_case
    else:
        candidates = [x for x in state['cases'] if x['status'] == 'passed']
        if not candidates:
            raise RuntimeError('No candidate completed the full update/probe/refresh path')
        # Keep the measured capacity boundary as well as the repeatable speed winner.
        total_mib = min(x[2] for x in gpu_state())
        stable = [x for x in candidates if max(x['nvml_peak_mib'].values()) <= total_mib * 0.95]
        if not stable:
            raise RuntimeError('No measured point leaves the required5% memory margin')
        fastest = min(stable, key=lambda x: x['mean_seconds'])
        winner = run_case(fastest['group_size'], fastest['micro_batch'], iterations=3, label='winner-repeat')
        if winner['status'] != 'passed' or max(winner['nvml_peak_mib'].values()) > total_mib * 0.95:
            raise RuntimeError('Winning configuration failed repeat/memory-margin acceptance')
        state['winner'] = winner
        state['status'] = 'complete'
    publish()
except BaseException as error:
    state['status'] = 'failed'
    state['error'] = f'{type(error).__name__}: {error}'
    publish()
    raise
