# -*- coding: UTF-8 -*-
import yaml
import xlrd
from time import time

from problems.DTLZ.DTLZ import *
from OREA.OREA import *

desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=4, suppress=True)


"""
OREA Tester for DTLZ benchmark functions.
"""
cfg_filename = 'config_DTLZ.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)
#config['event_file_name'] = './saved_surrogate/DTLZ150_lr0.01/'

name = 'DTLZ1'
dataset = DTLZ1(config)

# get the Pareto Front of DTLZ
pf_path = config['path_pf'] + name + " PF " + str(config['y_dim']) + "d "+str(5000)+".xlsx"
pf_data = xlrd.open_workbook(pf_path).sheets()[0]
n_rows = pf_data.nrows
pf = np.zeros((n_rows, config['y_dim']))
for index in range(n_rows):
    pf[index] = pf_data.row_values(index)

iteration_max = 1
for iteration in range(0, iteration_max):
    time1 = time()
    current_iteration = str(iteration + 1)#.zfill(2)
    alg = OREA(config, name, dataset, pf, random_init=True)
    alg.run(current_iteration)
    t = time() - time1
    print('run time:', t // 60, " mins, ", t % 60, " secs.")
    solution, minimum = alg.get_result()  # (b_save=False)
    print("solution: ", type(solution))
    print(solution)
    print("minimum: ", type(minimum))
    print(minimum)

