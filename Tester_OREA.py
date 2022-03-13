# -*- coding: UTF-8 -*-
import yaml
import xlrd
from time import time

from problems.DTLZ.DTLZ import *
#from OREA.OREA import *
from OREA.OREA_DACE import *
from tools.data_IO import load_PF

desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=4, suppress=True)


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-13.
OREA Tester for DTLZ benchmark functions.
"""
cfg_filename = 'config_DTLZ.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)

name = 'DTLZ1'
dataset = DTLZ1(config)

# get the Pareto Front of DTLZ
#"""
pf = load_PF(name)
"""
pf_path = config['path_pf'] + name + " PF " + str(config['y_dim']) + "d "+str(5000)+".xlsx"
pf_data = xlrd.open_workbook(pf_path).sheets()[0]
n_rows = pf_data.nrows
pf = np.zeros((n_rows, config['y_dim']))
for index in range(n_rows):
    pf[index] = pf_data.row_values(index)
#"""

iteration_max = 30
for iteration in range(0, iteration_max):
    time1 = time()
    current_iteration = str(iteration + 1).zfill(2)
    alg = OREA(config, name, dataset, pf, init_path='results/')
    alg.run(current_iteration)
    t = time() - time1
    print('run time:', t // 60, " mins, ", t % 60, " secs.")
    solution, minimum = alg.get_result()  # (b_save=False)
    print("solution: ", type(solution))
    print(solution)
    print("minimum: ", type(minimum))
    print(minimum)

