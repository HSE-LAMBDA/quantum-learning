import subprocess
from sklearn.model_selection import ParameterGrid
import os

param = {
    'projectors': [5, 10, 20, 40, 80],
    'measurements': [5, 10, 20, 40, 80]
}

param_grid = list(ParameterGrid(param))
for i in range(len(param_grid)):
    param_grid[i].update({'total' : param_grid[i]['projectors'] * param_grid[i]['measurements']})

print(len(param_grid), 'param combinations')

exp_name = 'exp'

if os.path.isfile(exp_name + '.results'):
    os.remove(exp_name + '.results')

for param_id, param_set in enumerate(param_grid):
    lst = ['python3', 'run.py']
    lst += ['exp_name', exp_name]
    lst += ['param_id', str(param_id)]
    for key in param_set:
        lst += [key, str(param_set[key])]
    
    while True:
        res = subprocess.run(lst)
        print('Return code:', res.returncode)
 
        if res.returncode == 0:
            break