# -*- coding: UTF-8 -*-
import numpy as np
from scipy import spatial
from time import time


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
labeling by pareto front
"""
def getPF(fitnesses, y):
    diff = fitnesses - y
    for i in range(len(diff)):
        if (diff[i] == 0).all():
            return fitnesses
    indexes = [index for index in range(len(fitnesses)) if min(diff[index]) < 0]
    new_fitnesses = fitnesses[indexes].copy()
    if min(np.max(diff, axis=1)) > 0:
        return np.append(new_fitnesses, y, axis=0)
    else:
        return new_fitnesses


# 8th June, 2019.  *** 8th Mutation, 9th PSO + mutation, 10th PSO.
# Revised 10th Oct, 2019 *** final version for OREA.
def domination_based_ordinal_values(pf_index, archive_fitness, pf_upperbound, pf_lowerbound,
                                    n_levels, overfitting_coeff=0.01, b_print=False):
    start = time()
    archive_size, n_objs = len(archive_fitness), len(archive_fitness[0])
    label = np.zeros(archive_size)
    cut_row = None  # a variable will be used in the second branch below, if the program enter the first branch, then this variable will always be None.
    # if all non-dominated solutions have the same objective values (Leading to a situation in which only one point exists in Pareto Front)
    if ((archive_fitness[pf_index] - pf_lowerbound) == 0).all():
        labeled_index_for_pf = [index for index in range(len(pf_index))]
        label[pf_index] = 1.0

        archive_fitness_0 = archive_fitness - pf_lowerbound
        nadir_upperbound = np.max(archive_fitness_0, axis=0)
        reference_point = np.array([nadir_upperbound/archive_size])
        if b_print:
            print("special case, only one point in current Pareto Front:", np.shape(reference_point))
    else:
        # check valid objs:
        valid_objs = [index for index in range(n_objs) if pf_upperbound[index] != pf_lowerbound[index]]
        archive_fitness = archive_fitness[:, valid_objs]
        pf_upperbound = pf_upperbound[valid_objs]
        pf_lowerbound = pf_lowerbound[valid_objs]
        n_objs = len(valid_objs)
        if b_print:
            print("valid_objs ", valid_objs)

        # select non-bound points. Points close to boundaries are sensitive to domination.
        labeled_index_for_pf = []
        archive_fitness_0 = archive_fitness - pf_lowerbound
        while len(labeled_index_for_pf) < 2:
            overfitting_bound = (pf_upperbound - pf_lowerbound) * overfitting_coeff
            for index_for_pf, index in enumerate(pf_index):
                if label[index] == 0:
                    status = archive_fitness_0[index] > overfitting_bound
                    if status.all():
                        labeled_index_for_pf.append(index_for_pf)
                        label[index] = 1.0
            overfitting_coeff -= 0.01
        reference_point = archive_fitness_0[pf_index[labeled_index_for_pf]]

        # shape the bounds
        shape_index_for_rp = []
        shape_reference = np.zeros((n_objs, n_objs))
        cut_row = np.zeros((n_objs))
        for i in range(n_objs):
            max_index_for_rp = np.argmax(reference_point[:, i])
            max_value = reference_point[max_index_for_rp, i]
            shape_index_for_rp.append(max_index_for_rp)
            shape_reference[i][i] = max_value
            cut_row[i] = max_value
        shape_index_for_rp = list(set(shape_index_for_rp))
        reference_point = np.delete(reference_point, shape_index_for_rp, axis=0)

        # include bound pf points which inside the new shape.
        for index_for_pf, index in enumerate(pf_index):
            if label[index] == 0 and min(np.max(shape_reference - archive_fitness_0[index], axis=1)) > 0:
                labeled_index_for_pf.append(index_for_pf)
                label[index] = 1.0

                reference_point = np.append(reference_point, archive_fitness_0[index].reshape(1, -1), axis=0)
        reference_point = np.append(reference_point, shape_reference, axis=0)
        if b_print:
            print("rp_upper_bound:", cut_row+pf_lowerbound)

    # Already labeled reference points, now prepare for labeling rest samples:
    rp_count = len(labeled_index_for_pf)
    rp_ratio = rp_count * 1.0 / archive_size
    if b_print:
        print("the number of shaped reference points: %d".format(rp_count))
        
    n_levels = max(int(np.ceil(1.0/rp_ratio)), n_levels)
    coeff = np.zeros((n_levels))

    value = 1.0  # 1.0 - pf_ratio
    value_diff = value/(n_levels-1)
    ratio_bound = rp_ratio
    ratio_diff = (1.0 - ratio_bound)/(n_levels-1)

    coeff_t = 1.0
    coeff[0] = coeff_t
    if b_print:
        print("%d: labeled ratio: %.3f < %.3f < %.3f. coff: %.3f, value: %.3f" % (1, 0.0, rp_ratio, ratio_bound, 1.0, value))
    rest_index = np.array([i for i in range(archive_size) if label[i] == 0])
    rest_fitness = archive_fitness_0[rest_index]
    rp_coeff = np.zeros((len(rest_index)))
    if cut_row is None:
        rp_coeff = np.nanmin(rest_fitness/reference_point[0], axis=1)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(len(rest_index)):
                min_col = np.nanmin(rest_fitness[i]/(reference_point[: -n_objs]), axis=1)
                rp_coeff[i] = max(np.append(min_col, rest_fitness[i]/cut_row, axis=0))
    rp_coeff_order = np.argsort(rp_coeff)
 
    start_index = 0
    for c in range(1, n_levels-1):
        ratio = int(min((np.ceil((rp_ratio+c * ratio_diff) * archive_size))-rp_count, len(rp_coeff_order)-1))
        value -= value_diff
        labeling_indexes = rest_index[rp_coeff_order[start_index: ratio]]
        label[labeling_indexes] = value
        coeff[c] = rp_coeff[rp_coeff_order[ratio]]
        if b_print:
            print("%d: labeled ratio: %.3f to %.3f. coff: %.3f, value: %.3f" % (c+1, (start_index+rp_count)*1.0/archive_size, (ratio+rp_count)*1.0/archive_size, coeff[c], value))
        start_index = ratio
    if b_print:
        print(label[-2:], "time for labeling operation: %.5f" % (time()-start))

    #print("the number of ordinal levels %d." % n_levels)
    #print("indexes of reference points for current pf: ", labeled_index_for_pf)
    return label, n_levels, reference_point, labeled_index_for_pf


