"""Run the model jobs of a run as separate processes on local GPUs, one queue slot per GPU and slot.

Training jobs train and evaluate every source of every domain, Base included (27 jobs). Inference jobs replay the common
test with the DP model of each domain and answer the preference-weighted test with the four main-bank models of each
domain (15 jobs). A failed job is reported and the queue continues; rerun a job only after a resource failure.

Usage:
  python scripts/run_models.py --config CONFIG --python MODEL_PYTHON --gpus 0,1,2,3 --slots-per-gpu 2
  python scripts/run_models.py --config CONFIG --python MODEL_PYTHON --gpus 0,1,2,3 --inference
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from privacyqa.common import DOMAINS  # noqa: E402
from privacyqa.config import Config  # noqa: E402
from privacyqa.qa.train import SOURCES  # noqa: E402


def jobs(args):
    domains = args.domains.split(',') if args.domains else DOMAINS
    if not args.inference:
        return [(f'{d}-{m}', ['privacyqa.qa.train', '--domain', d, '--method', m]) for d in domains for m in SOURCES]
    result = []
    for d in domains:
        result.append((f'replay-{d}-DP', ['privacyqa.qa.infer', '--mode', 'replay', '--domain', d, '--method', 'DP']))
        result.extend((f'{d}-{m}', ['privacyqa.qa.infer', '--mode', 'preference', '--domain', d, '--method', m]) for m in SOURCES[1:5])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--python', required=True, help='interpreter of the model environment')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--slots-per-gpu', type=int, default=1)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.45, help='bounds the memory of each job; results do not depend on it')
    parser.add_argument('--domains', help='comma-separated subset of domains')
    parser.add_argument('--inference', action='store_true', help='run the replays and the preference-weighted test')
    args = parser.parse_args()
    config = Config.load(args.config)
    logs = config.stage('preference-logs' if args.inference else 'qa-logs')
    logs.mkdir(parents=True, exist_ok=True)
    queue, lock, results, started = Queue(), threading.Lock(), [], time.monotonic()
    for job in jobs(args):
        queue.put(job)

    def worker(gpu):
        while True:
            try:
                name, command = queue.get_nowait()
            except Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(REPO / 'src'), TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
            with (logs / f'{name}.log').open('x') as stream:
                code = subprocess.run([args.python, '-m', *command, '--config', str(config.path), '--device', 'cuda:0',
                                       '--gpu-memory-fraction', str(args.gpu_memory_fraction)],
                                      cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            record = dict(job=name, gpu=gpu, exit_code=code, elapsed_seconds=round(time.monotonic() - started, 1))
            with lock:
                results.append(record)
                print(json.dumps(record), flush=True)

    gpus = args.gpus.split(',')
    with ThreadPoolExecutor(max_workers=len(gpus) * args.slots_per_gpu) as pool:
        for future in [pool.submit(worker, gpu) for _ in range(args.slots_per_gpu) for gpu in gpus]:
            future.result()
    failed = [r['job'] for r in results if r['exit_code']]
    print(json.dumps(dict(completed=len(results) - len(failed), failed=failed)))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
