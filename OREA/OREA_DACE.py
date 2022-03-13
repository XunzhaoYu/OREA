# -*- coding: UTF-8 -*-
# --- basic libraries ---
import numpy as np
from scipy import spatial
import xlrd
import xlwt
from time import time
from copy import deepcopy
# --- surrogate modeling ---
from models.pydacefit.dace import *
from models.pydacefit.regr import *
from models.pydacefit.corr import *
# --- OREA ---
from OREA.reference_vector import generate_vectors
from OREA.labeling_operator import domination_based_ordinal_values
from inspyred import ec
from random import Random
# --- optimization libraries ---
from optimization.operators.crossover_operator import *
from optimization.operators.mutation_operator import *
from optimization.operators.selection_operator import *
from optimization.PSO import *
from optimization.EI import *
from optimization.performance_indicators import *
# --- tools ---
from tools.recorder import *

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-13.
SSCI version OREA, use Kriging (DACEfit), has a higher computational efficiency than OREA.py.

X. Yu, X. Yao, Y. Wang, L. Zhu, and D. Filev, “Domination-based ordinal regression for expensive multi-objective optimization,” 
in Proceedings of the 2019 IEEE Symposium Series on Computational Intelligence (SSCI’19), 2019, pp. 2058–2065.
"""


class OREA:
    def __init__(self, config, name, dataset, pf, init_path=None):
        self.config = deepcopy(config)
        self.init_path = init_path
        # --- problem setups ---
        self.name = name
        self.n_vars = self.config['x_dim']
        self.n_objs = self.config['y_dim']
        self.upperbound = np.array(self.config['x_upperbound'])
        self.lowerbound = np.array(self.config['x_lowerbound'])
        self.dataset = dataset
        self.true_pf = pf
        self.indicator_IGD_plus = inverted_generational_distance_plus(reference_front=self.true_pf)
        self.indicator_IGD = inverted_generational_distance(reference_front=self.true_pf)

        # --- surrogate setups ---
        self.n_levels = self.config['n_levels']
        self.dace_training_iteration_init = self.config['dace_training_iteration_init']
        self.dace_training_iteration = self.config['dace_training_iteration']
        self.coe_range = self.config['coe_range']
        self.exp_range = self.config['exp_range']

        # --- optimization algorithm setups ---
        self.evaluation_init = self.config['evaluation_init']
        self.evaluation_max = self.config['evaluation_max']

        self.n_reproduction = 2
        self.n_gen_reproduction = 1
        self.search_evaluation_max = self.config['search_evaluation_max']
        self.pop_size = self.config['population_size']
        self.neighborhood_size = self.config['neighborhood_size']
        self.n_variants = self.config['n_variants']

        # --- --- reference vectors --- ---
        self.vectors = generate_vectors(self.n_objs, layer=3, h=2, h2=1)
        self.normalized_vs = self.vectors / np.sqrt(np.sum(np.power(self.vectors, 2), axis=1)).reshape(-1, 1)
        self.n_vectors = len(self.vectors)
        # --- --- crossover operator --- ---
        self.crossover_args = self.config['crossover_args']
        crossover_ops = {
            'SBX': SBX(self.crossover_args[0], self.crossover_args[1])
        }
        self.crossover_op = crossover_ops[self.config['crossover_op']]
        # --- --- mutation operator --- ---
        self.mutation_args = self.config['mutation_args']
        mutation_ops = {
            'polynomial': Polynomial(self.mutation_args[0], self.mutation_args[1]),
            'value_shift': ValueShift(self.mutation_args[0])
        }
        self.mutation_op = mutation_ops[self.config['mutation_op']]

        # --- variables declarations ---
        self.time = None
        self.iteration = None
        # --- --- surrogate and archive variables --- ---
        self.theta = np.zeros((2*self.n_vars))  # parameters of the ordinal regression surrogates
        self.surrogate = None
        self.X = None
        self.Y = None
        self.archive_size = 0
        self.nadir_upperbound = None  # upperbound of Y.
        # --- --- pareto front variables --- ---
        self.pf_index = None  # the indexes of pareto set solutions in the archive.
        self.ps = None  # pareto set: decision space
        self.pf = None  # pareto front: objective space
        self.pf_upperbound = self.pf_lowerbound = None  # pf boundaries
        
        self.objective_range = None  # self.nadir_upperbound - self.pf_lowerbound
        self.normalized_pf = None
        self.pf_changed = self.range_changed =  None  # update flags
        self.miss_counter = 0
        # --- labeling methods ---
        self.label = self.reference_point = self.rp_index_for_pf = None
        self.region_id = self.subspace_pf_counter = self.rp_subspace_indexes = self.candidate_subspace_indexes = None
        # --- recorder ---
        self.performance = np.zeros(2)
        self.recorder = None

    """
    Initialization methods:
    set variables and surrogate for a new iteration.
    """
    def variable_init(self, current_iteration):
        """
        Initialize surrogate, reset all variables.
        """
        self.time = time()
        self.iteration = current_iteration
        # --- surrogate and archive variables ---
        self.theta = np.append(np.ones(self.n_vars) * np.mean(self.coe_range), np.ones(self.n_vars) * np.mean(self.exp_range))
        self.X, self.Y = self._archive_init()
        self.archive_size = len(self.X)
        self.nadir_upperbound = np.max(self.Y, axis=0)
        # --- pareto front variables ---
        self.pf_lowerbound = np.ones((self.n_objs,)) * float('inf')
        self.pf_index = np.zeros(1, dtype=int)
        self.ps, self.pf = self.ps_init()
        self.pf_upperbound = np.max(self.pf, axis=0)
        print("Initialization of non-dominated solutions:", np.shape(self.ps))
        print("Initial Pareto Front:")
        print(self.pf)
        self.objective_range = self.nadir_upperbound - self.pf_lowerbound
        self.objective_range[self.objective_range == 0] += 0.0001  # avoid NaN caused by dividing zero.
        print("Objective range:", self.objective_range)
        self.normalized_pf = (self.pf - self.pf_lowerbound) / self.objective_range  
        # --- --- update flags --- ---
        self.pf_changed = True
        self.range_changed = True
        self.miss_counter = 0.0
        # --- labeling methods ---
        self.label = None  #np.zeros((1, self.archive_size))
        self.reference_point = np.zeros((1, self.n_objs))
        self.rp_index_for_pf = []  # indexes in pf.

        self.region_id = np.ones((self.archive_size,), dtype=int) * -1
        self.subspace_pf_counter = []
        self.rp_subspace_indexes = []  # the index list of subspaces which contain Reference Points.
        # local: self.non_empty_subspace_indexes  # the index list of subspaces with at least one pf points
        self.candidate_subspace_indexes = []  # the index list of subspaces with the least pf points, should be updated once pf is changed.
        # --- recorder ---
        self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
        self.performance[1] = self.indicator_IGD.compute(self.pf)
        print("Initial IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))
        self.recorder = Recorder(self.name)
        self.recorder.init(self.X, self.Y, self.performance, ['IGD+', 'IGD'])
        if self.init_path is None:
            path = self.config['path_save'] + self.name + "/Initial(" + str(self.n_vars) + "," + str(self.n_objs) + ")/" + \
                   str(self.evaluation_init) + "_" + self.iteration + ".xlsx"
            self.recorder.save(path)
        

    # Invoked by self.variable_init()
    def _archive_init(self):
        """
        Modify this method to initialize your 'self.surrogate'.
        :param b_exist: if surrogate is existing. Type: bool.
        :return X: initial samples. Type: 2darray. Shape: (self.evaluation_init, self.n_vars)
        :return Y: initial fitness. Type: 2darray. Shape: (self.evaluation_init, self.n_objs)
        """
        if self.init_path is None:
            X, Y = self.dataset.sample(n_samples=self.evaluation_init)
        else:  # load pre-sampled dataset
            path = self.init_path + self.name + "/Initial(" + str(self.n_vars) + "," + str(self.n_objs) + ")/" + \
                   str(self.evaluation_init) + "_" + self.iteration + ".xlsx"
            src_file = xlrd.open_workbook(path)
            src_sheets = src_file.sheets()
            src_sheet = src_sheets[0]
            X = np.zeros((self.evaluation_init, self.n_vars), dtype=float)
            Y = np.zeros((self.evaluation_init, self.n_objs), dtype=float)
            for index in range(self.evaluation_init):
                row_data = src_sheet.row_values(index + 1)
                X[index] = row_data[1:1 + self.n_vars]
                Y[index] = row_data[1 + self.n_vars:1 + self.n_vars + self.n_objs]
            Y = np.around(Y, decimals=4)
        return X, Y

    """
    Pareto Set/Front methods
    """
    def ps_init(self):
        ps = np.array([self.X[0]])
        pf = np.array([self.Y[0]])
        for index in range(1, self.archive_size):
            ps, pf = self.get_ps(ps, pf, np.array([self.X[index]]), np.array([self.Y[index]]), index)
        return ps, pf

    def get_ps(self, ps, pf, x, y, index):
        diff = pf - y
        diff = np.around(diff, decimals=4)
        # --- check if y is the same as a point in pf (x is not necessary to be the same as a point in ps) ---
        # --- 检查新的点是否在pf上的一点相同 (obj space上相同不代表decision space上也相同) ---
        for i in range(len(diff)):
            if (diff[i] == 0).all():
                self.pf_index = np.append(self.pf_index, index)
                self.pf_changed = True
                return np.append(ps, x, axis=0), np.append(pf, y, axis=0)
        # --- update nadir objective vector (upperbound) ---
        for obj in range(self.n_objs):
            if self.nadir_upperbound[obj] < y[0][obj]:
                self.nadir_upperbound[obj] = y[0][obj]
                self.range_changed = True
        # exclude solutions (which are dominated by new point x) from the current PS. # *** move to if condition below? only new ps point can exclude older ones.
        index_newPs_in_ps = [index for index in range(len(ps)) if min(diff[index]) < 0]
        self.pf_index = self.pf_index[index_newPs_in_ps]
        new_pf = pf[index_newPs_in_ps].copy()
        new_ps = ps[index_newPs_in_ps].copy()
        # --- add new point x into the current PS, update PF ---
        if min(np.max(diff, axis=1)) > 0:
            self.pf_index = np.append(self.pf_index, index)
            self.pf_changed = True
            # update ideal objective vector (lowerbound):
            for obj in range(self.n_objs):
                if self.pf_lowerbound[obj] > y[0][obj]:
                    self.pf_lowerbound[obj] = y[0][obj]
                    self.range_changed = True
            self.miss_counter = 0.0
            return np.append(new_ps, x, axis=0), np.append(new_pf, y, axis=0)
        else:
            self.miss_counter += 1.0
            return new_ps, new_pf

    """
    Evaluation on real problem.
    """
    def _population_evaluation(self, population, is_normalized_data=False, upperbound=None, lowerbound=None):
        if is_normalized_data:
            population = population*(upperbound-lowerbound)+lowerbound
        fitnesses = self.dataset.evaluate(population)
        return np.around(fitnesses, decimals=4)

    """
    Main method
    """
    def run(self, current_iteration):
        self.variable_init(current_iteration)
        while self.archive_size < self.evaluation_max:
            """
            if (self.archive_size - self.evaluation_init) % 10 == 0:
                self.recorder.save("Temp-" + self.name + "-" + self.iteration + ".xlsx")
            """
            print(" ")
            print(" --- Labeling and Training Kriging model... --- ")
            self.label = np.zeros(self.archive_size)
            last_n_levels = self.n_levels
            self.label, self.n_levels, self.reference_point, self.rp_index_for_pf = \
                domination_based_ordinal_values(self.pf_index, self.Y, self.pf_upperbound, self.pf_lowerbound, self.n_levels, overfitting_coeff=0.03, b_print=False)

            self.surrogate = DACE(regr=regr_constant, corr=corr_gauss2, theta=self.theta,
                             thetaL=np.append(np.ones(self.n_vars) * self.coe_range[0], np.ones(self.n_vars) * self.exp_range[0]),
                             thetaU=np.append(np.ones(self.n_vars) * self.coe_range[1], np.ones(self.n_vars) * self.exp_range[1]))
            if self.n_levels == last_n_levels:
                self.surrogate.fit(self.X, self.label, self.dace_training_iteration)
            else:
                self.surrogate.fit(self.X, self.label, self.dace_training_iteration_init)
            self.theta = self.surrogate.model["theta"]
            print("updated theta:", self.theta)

            print(" --- Reproduction: searching for minimal negative EI... --- ")
            self.new_point = np.zeros((self.n_reproduction, self.n_vars))
            if len(self.pf_index) == 1:
                for i in range(self.n_reproduction):
                    self.new_point[i] = self._reproduce_by_one_mutation(self.X[self.pf_index[0]], times_per_gene=self.n_variants, miss=int(self.miss_counter))
            else:
                new_point_pre = self._reproduce_by_PSO(self.n_gen_reproduction)
                self.new_point[0] = new_point_pre[0]

                print(" --- --- IndReproduction: mating 1: --- --- ")
                if self.pf_changed:
                    self._get_region_ID(self.normalized_pf)
                    print("pf id:", self.region_id)

                    # record indexes of the subspaces which contain level0 points(reference points).
                    self.rp_subspace_indexes = []
                    for i in self.rp_index_for_pf:
                        self.rp_subspace_indexes.append(self.region_id[i])
                    self.rp_subspace_indexes = np.array(list(set(self.rp_subspace_indexes)))
                    print("rp subspace id:", self.rp_subspace_indexes)

                    # 每个子空间有几个pf点
                    self.subspace_pf_counter = np.zeros((self.n_vectors), dtype=int)
                    for i in range(len(self.region_id)):
                        self.subspace_pf_counter[self.region_id[i]] += 1

                    # delete subspaces without pf points: all subspace indexes -> non_empty_subspace_indexes
                    n_points_order = np.argsort(self.subspace_pf_counter)  # rank: min -> max
                    min_n_points, min_n_index = 0, 0
                    for index, rank in enumerate(n_points_order):
                        if self.subspace_pf_counter[rank] > 0:
                            min_n_points = self.subspace_pf_counter[rank]
                            min_n_index = index
                            break
                    self.non_empty_subspace_indexes = n_points_order[min_n_index:]

                    # select subspaces with the least PF points: non_empty_subspace_indexes -> candidate_subspace_indexes
                    self.candidate_subspace_indexes = []
                    for subspace_index in self.non_empty_subspace_indexes:
                        if self.subspace_pf_counter[subspace_index] > min_n_points:
                            break
                        self.candidate_subspace_indexes.append(subspace_index)

                target_subspace = np.random.choice(self.candidate_subspace_indexes, 1)[0]  # 第 target 个子空间有最少的pf点,最少为1.
                print("target subspace:", target_subspace, "in", self.candidate_subspace_indexes, ":", self.vectors[target_subspace])

                pf_index_in_subspace = [s for s in range(len(self.region_id)) if self.region_id[s] == target_subspace]  # subspace pf points: index in pf_index
                target_pf_index = self.pf_index[
                    pf_index_in_subspace]  # [self.pf_index[s] for s in pf_index_in_subspace]  # subspace pf points: index in archive
                print("PF indexes in target subspace:", target_pf_index)

                # select point with maximal label value in subspace: crossover operator added.
                max_value = np.max(self.label[target_pf_index])
                candidate_indexes = [ind for ind in pf_index_in_subspace if self.label[self.pf_index[ind]] == max_value]  # index for pf_index
                # select based on crowd
                candidate_distance = spatial.distance.cdist(self.normalized_pf[candidate_indexes], self.normalized_pf[candidate_indexes])
                candidate_distance += np.eye(len(candidate_indexes)) * float('inf')

                mating1_index = self.pf_index[candidate_indexes[np.argmax(np.min(candidate_distance, axis=1), axis=0)]]

                mating_population = np.zeros((2, self.n_vars))
                mating_population[0] = self.X[mating1_index]
                print("mating 1:", mating1_index, mating_population[0], self.Y[mating1_index])

                print(" --- --- IndReproduction: mating 2: --- --- ")
                # """
                random_subspace = target_subspace
                random_candidate_indexes = []
                if len(self.rp_subspace_indexes) == 1 and self.rp_subspace_indexes[0] == target_subspace:  # reference points 全在 target 子空间
                    random_subspace = np.random.choice(self.candidate_subspace_indexes, 1)[0]
                    for i, id in enumerate(self.region_id):
                        if id == random_subspace:
                            random_candidate_indexes.append(self.pf_index[i])
                else:
                    while random_subspace == target_subspace:
                        random_subspace = np.random.choice(self.rp_subspace_indexes, 1)[0]
                    for i, id in enumerate(self.region_id[self.rp_index_for_pf]):
                        if id == random_subspace:
                            random_candidate_indexes.append(self.pf_index[self.rp_index_for_pf[i]])
                print("random subspace:", random_subspace, "RP indexes in random subspace", random_candidate_indexes)
                mating2_index = np.random.choice(random_candidate_indexes, 1)[0]
                mating_population[1] = self.X[mating2_index]
                print("mating 2:", mating2_index, mating_population[1], self.Y[mating2_index])

                local_origin = self.crossover_op.execute(mating_population, self.upperbound, self.lowerbound)
                #"""
                if random() < 0.5:
                    local_origin = local_origin[0]
                else:
                    local_origin = local_origin[1]
                # """
                #self.new_point[0] = self._reproduce_by_one_mutation(local_origin[0], times_per_gene=self.n_variants, miss=int(self.miss_counter))
                self.new_point[1] = self._reproduce_by_one_mutation(local_origin, times_per_gene=self.n_variants, miss=int(self.miss_counter))

            # end of selection process.
            self.new_point_objs = self._population_evaluation(self.new_point, True, self.upperbound, self.lowerbound)
            print(" --- Evaluate on fitness function... ---")
            print("new point:", self.new_point)
            print("new point objective ", self.new_point_objs)
            # --- update archive, archive_fitness, distance in model ---
            self.X = np.append(self.X, self.new_point, axis=0)
            self.Y = np.append(self.Y, self.new_point_objs, axis=0)
            self.archive_size += self.n_reproduction

            # after used to initialize the kriging model, the archive then used to save Pareto optimal solutions
            self._progress_update()


    def _reproduce_by_PSO(self, n_selected_population=1):
        ea = PSO(Random())
        ea.terminator = no_improvement_termination
        ea.topology = inspyred.swarm.topologies.ring_topology
        final_pop = ea.evolve(generator=generate_population,
                              evaluator=self.cal_EI_for_inspyred,
                              pop_size=self.pop_size,
                              maximize=False,
                              bounder=ec.Bounder(self.lowerbound, self.upperbound),
                              max_evaluations=self.search_evaluation_max,
                              neighborhood_size=self.neighborhood_size,
                              num_inputs=self.n_vars)
        final_pop.sort(reverse=True)  # minimal first when minimize, maximal first when maximize
        selected_population = np.zeros((n_selected_population, self.n_vars))
        for i, ind in enumerate(final_pop[:n_selected_population]):
            selected_population[i] = ind.candidate
        return selected_population

    def _reproduce_by_one_mutation(self, origin, times_per_gene=100, miss=0):
        neg_ei = np.zeros((self.n_vars * times_per_gene))
        new_point = np.tile(origin.copy(), (self.n_vars * times_per_gene, 1))

        mutant = self.mutation_op.execute(new_point, self.upperbound, self.lowerbound, unique=True)
        for i in range(self.n_vars * times_per_gene):
            neg_ei[i] = self.cal_EI(mutant[i])
        return mutant[np.argmin(neg_ei)].copy()
        """  # the mechanism of miss_counter is deleted to simplify OREA
        if miss < 1:
            return mutant[np.argmin(neg_ei)].copy()
        else:
            order = np.argsort(neg_ei)
            miss = min(self.n_vars - 1, miss)
            selected = np.random.choice(order[:miss * times_per_gene], 1)[0]
            new_point = mutant[selected].copy()
            return new_point
        """

    def cal_EI_for_inspyred(self, candidates, args):
        fitness = []
        for ind in candidates:
            f = self.cal_EI(ind)
            fitness.append(f)
        return fitness

    def cal_EI(self, x):  # minimize negative EI equivalent to maximize EI.
        x = np.array(x).reshape(1, -1)
        mu_hat, sigma2_hat = self.surrogate.predict(x, return_mse=True)
        if sigma2_hat <= 0.:
            ei = mu_hat - 1.0
        else:  # cdf(z) = 1/2[1 + erf(z/sqrt(2))].
            ei = EI(minimum=-1.0, mu=-mu_hat, sigma=np.sqrt(sigma2_hat))
        return -ei


    def _get_region_ID(self, region_points, incremental=False):
        projection_length = self.normalized_vs.dot(region_points.T)  # n_vectors * archive_size
        region_id = np.ones((len(region_points)), dtype=int) * -1
        for i in range(len(region_points)):
            region_distance_vector = (region_points[i].reshape(1, -1) - projection_length[:, i].reshape(-1, 1) * self.normalized_vs)
            region_distance = np.sum(np.power(region_distance_vector, 2), axis=1)
            region_id[i] = np.argmin(region_distance)
        if incremental:
            self.region_id = np.append(self.region_id, region_id, axis=0)
        else:
            self.region_id = region_id

    def _progress_update(self):
        self.pf_changed, self.range_changed = False, False
        for new_index in range(self.n_reproduction):
            index = self.archive_size - self.n_reproduction + new_index
            self.ps, self.pf = self.get_ps(self.ps, self.pf, np.array([self.new_point[new_index]]), np.array([self.new_point_objs[new_index]]), index)
            if self.pf_changed:
                self.performance[0] = self.indicator_IGD_plus.compute(self.pf)
                self.performance[1] = self.indicator_IGD.compute(self.pf)
            self.recorder.write(index+1, self.new_point[new_index], self.new_point_objs[new_index], self.performance)
        print("update archive to keep all individuals non-dominated. ", np.shape(self.ps))

        # update three bounds for pf, and also update normalized pf
        if self.range_changed:
            self.objective_range = self.nadir_upperbound-self.pf_lowerbound
            self.objective_range[self.objective_range == 0] =+ 0.0001

        if self.pf_changed:
            print("pf_index", self.pf_index)
            print("pf", self.pf)
            self.pf_upperbound = np.max(self.pf, axis=0)
            print("pf upper bound:", self.pf_upperbound)

        if self.range_changed or self.pf_changed:
            self.normalized_pf = (self.pf-self.pf_lowerbound)/self.objective_range

        # print results
        t = time() - self.time
        print("OREA, Evaluation Count: {:d}.  Total time: {:.0f} mins, {:.2f} secs.".format(self.archive_size, t // 60, t % 60))
        print("Current IGD+ value: {:.4f}, IGD value: {:.4f}.".format(self.performance[0], self.performance[1]))

    def get_result(self):
        path = self.config['path_save'] + self.name + "/Total(" + str(self.n_vars) + "," + str(self.n_objs) + ")/" + \
               str(self.evaluation_max) + "_" + self.iteration + " igd " + str(np.around(self.performance[1], decimals=4)) + ".xlsx"
        self.recorder.save(path)
        return self.ps, self.performance[0]
