import os
import sys
import json
import subprocess
from sklearn.model_selection import ParameterGrid

exp_name = 'raise_dim'
exp_path = os.path.join('exps', exp_name)
run_path = os.path.join(exp_path, 'run.py')
config_path = os.path.join(exp_path, 'config.json')
assert os.path.isdir(exp_path), 'Experiment {} do not exists'.format(exp_path)
assert os.path.isfile(run_path), "There is no run.py file in {}".format(exp_path)
assert os.path.isfile(config_path), "There is no config.json file in {}".format(exp_path)

with open(config_path) as config_file:
    param = json.load(config_file)

param_grid = list(ParameterGrid(param))
for i in range(len(param_grid)):
    param_grid[i].update({'total': param_grid[i]['projectors'] * param_grid[i]['measurements']})

print(len(param_grid), 'param combinations')


if os.path.isfile(os.path.join(exp_name, 'results.txt')):
    os.remove(os.path.join(exp_name, 'results.txt'))

for param_id, param_set in enumerate(param_grid):
    lst = [sys.executable, run_path]
    lst += ['exp_path', exp_path]
    lst += ['param_id', str(param_id)]
    for key in param_set:
        lst += [key, str(param_set[key])]

    res = subprocess.run(lst)
    print('Return code:', res.returncode)
    break
