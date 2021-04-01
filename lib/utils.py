import numpy as np
import pandas as pd
import linecache
from scipy.linalg import sqrtm
from .simulator import projectorOnto, measure
import os
import subprocess as sp
import psutil
import re

def generate_projectors(dim:int, projectors_cnt:int, noise:float=0):
    projectors_orig, projectors_noised = [], []  # E
    for _ in range(projectors_cnt):
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        proj = projectorOnto(psi)
        proj /= np.trace(proj)
        projectors_orig.append(proj)

        psi = psi + np.random.normal(0, noise)
        proj = projectorOnto(psi)
        proj /= np.trace(proj)
        projectors_noised.append(proj)
        
    return np.array(projectors_orig), np.array(projectors_noised)


def generate_y(rho, train_X, measurements_cnt):
    train_y = []
    for proj in train_X:
        ones_cnt, _ = measure(measurements_cnt, rho, proj)
        train_y.append(ones_cnt / measurements_cnt)
    return np.array(train_y).astype('float64')

def generate_dataset(target_state, projectors_cnt, measurements_cnt, noise:float=0):
    dim = target_state.shape[0]
    train_X_orig, train_X_noised = generate_projectors(dim, projectors_cnt, noise)
    train_y = generate_y(target_state, train_X_orig, measurements_cnt)
    return train_X_orig, train_X_noised, train_y


def fidelity(state_1, state_2):
    state_1_sqrt = sqrtm(state_1)
    F = np.dot(np.dot(state_1_sqrt, state_2), state_1_sqrt)
    return (np.trace(sqrtm(F)) ** 2).real


def read_data(path):
    target_state_string = linecache.getline(path, 3)[16:-2]
    target_state_raw = np.array(target_state_string.split(), dtype=np.float32)
    target_state = projectorOnto(target_state_raw[::2] + 1j * target_state_raw[1::2])

    df = pd.read_csv(path, skiprows=4, delim_whitespace=True, header=None)
    total_ticks = df[0].values[-1]
    train_y = df[1].values * df.shape[0] / (total_ticks*target_state.shape[0])

    psi_list = df.iloc[:, 2:].values[:, ::2] + 1j * df.iloc[:, 2:].values[:, 1::2]
    train_X = []
    for psi in psi_list:
        proj = projectorOnto(psi)
        train_X.append(proj)
    return target_state, np.array(train_X), train_y


def get_gpu_memory_usage(pid=None):
    if pid is None: pid=os.getpid()
    output = sp.run(['nvidia-smi'], stdout=sp.PIPE).stdout.decode()
    for line in output.split('\n'):
        res = re.findall(f'.*\d*\s*{pid}\s*[CG]\s*.*python\s*(\d*)MiB.*', line)
        if res: return int(res[0]) # Mb


def get_ram_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in Mb