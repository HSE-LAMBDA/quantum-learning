import os
import sys
import json
import tempfile
import argparse
import itertools
from multiprocessing.pool import ThreadPool
from subprocess import STDOUT, call
ROOT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

thread_limit = 3
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parsed_args = parser.parse_args()

with open(parsed_args.config) as f:
    exps = json.load(f)

processes = []
log_dir = os.path.join(ROOT_PATH, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = open(os.path.join(log_dir, 'results'), 'a+')

commands = []
for exp in exps['params']:
    cmd = [sys.executable, exps['script']]
    cmd += [str(x) for x in itertools.chain(*exp.items())]
    commands.append(cmd)


def run(cmd):
    f = tempfile.NamedTemporaryFile(mode='w+', dir=log_dir, delete=False)
    return cmd, f, call(cmd, stdout=f, stderr=STDOUT)

for cmd, f, rc in ThreadPool(thread_limit).imap_unordered(run, commands):
    if rc != 0:
        print('{cmd} failed with exit status: {rc}'.format(**vars()))

    processes.append((cmd, f))

for _, f in processes:
    f.seek(0)
    log_file.write(f.read())
    f.close()

log_file.close()