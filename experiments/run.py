import subprocess
from multiprocessing import Pool
import os
import json
import random

N_WORKERS = 5
RESULTS_DIR = 'results/'

def run_task(task):
    if os.path.isfile(f"{RESULTS_DIR}/{task['id']}.json"):
        pass
    else:
        cmd = ['python', 'main.py', str(task['id']), str(task['executions_cnt']),
                    str(task['algorithm']), '--n_qubits', str(task['n_qubits']), '--data_type', str(task['data_type'])]
        if 'file' in task: cmd = cmd + ['--file', task['file']]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()

if __name__ == '__main__':
    with open('tasks.json', 'r') as f:
        tasks = json.load(f)
    random.shuffle(tasks)
    pool = Pool(N_WORKERS)
    pool.map(run_task, tasks)