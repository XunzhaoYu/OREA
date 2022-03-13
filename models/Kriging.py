# -*- coding: UTF-8 -*-
import numpy as np
from scipy.linalg import solve, cholesky, det, pinv, inv
import inspyred
from inspyred import ec
from random import Random
from copy import deepcopy

from optimization.PSO import *
from optimization.operators.mutation_operator import *

desired_width = 250
np.set_printoptions(linewidth=desired_width)

""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Last update: 2022-Mar-01.
A Kriging model trained by PSO. Equations appear in the code are available in:
D. R. Jones, M. Schonlau, and W. J. Welch, “Efficient global optimiza- tion of expensive black-box functions,” 
Journal of Global Optimization, vol. 13, no. 4, pp. 455–492, 1998.
"""


class Kriging:
    def __init__(self, config, data):
        # prior data
        self.n_vars = config['x_dim']
        self.n_objs = config['y_dim']
        coe_range = config['coe_range']
        exp_range = config['exp_range']
        self.pop_size = config['model_training_population_size']
        self.neighborhood_size = config['model_training_neighborhood_size']

        self.label = None  # numpy.ndarray 1D, archive_size
        self.archive_size = len(data)
        self.distance = np.zeros((self.archive_size, self.archive_size, self.n_vars))  # lower triangle matrix
        for i in range(1, self.archive_size):
            for j in range(i):
                self.distance[i, j] = np.abs((data[i] - data[j]))

        # model status
        self.R = np.zeros((self.archive_size, self.archive_size), dtype=np.float)
        self.U = None
        self.mu = 0
        self.sigma = 0
        self.prediction_y_temp = 0  # = solve(self.U, solve(self.U.T, fitness - one * self.mu))

        # boundary of kriging parameters
        self.param_upperbound = np.append(np.ones((self.n_vars,)) * coe_range[1], np.ones((self.n_vars,)) * exp_range[1])
        self.param_lowerbound = np.append(np.ones((self.n_vars,)) * coe_range[0], np.ones((self.n_vars,)) * exp_range[0])
        # parameters initialization
        self.param = np.append(np.ones((self.n_vars,)) * coe_range[1], np.ones((self.n_vars,)) * exp_range[1])

    def train(self, label, predefine=False, max_evaluation=5000):
        self.label = label.copy()
        print("archive size: ", self.archive_size)
        if predefine:
            self.param = deepcopy(self.param_upperbound)
        else:
            self.minNegLnLik = self.likelihood(self.param)[0]  # minimal negative ln(likelihood), equivalents to maximal positive ln(likelihood).
            print("before training: parameters, ", self.param, ".  likelihood, ", self.minNegLnLik)

            # Using PSO to optimize the hyper-parameters in the Kriging model.
            ea = PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(
                generator=self.generate_population,
                evaluator=self.likelihood_for_param,
                pop_size=self.pop_size,
                maximize=False,
                bounder=ec.Bounder(self.param_lowerbound, self.param_upperbound),
                max_evaluations=max_evaluation,
                neighborhood_size=self.neighborhood_size,
                num_inputs=self.n_vars*2)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)
            # final_pop is a list of Individual

            """
            updated on 16th May, 2019: three lines.
            check if new parameters and likelihood are better than previous one. # or using warming up during the optimization of parameters in the future.
            """
            print("negative log likelihood：", self.minNegLnLik, final_pop[0].fitness)
            if self.minNegLnLik > final_pop[0].fitness:
                self.param = final_pop[0].candidate
                self.minNegLnLik = final_pop[0].fitness

            print("after training parameters: ", self.param, ". likelihood", self.minNegLnLik)
            # """  # end of inspyred

        # write training results:
        self.R, self.U, self.mu, self.sigma, LnDetPsi = self.likelihood(self.param)[1:]
        # !!! for speeding up prediction processes !!!
        self.prediction_y_temp = solve(self.U, solve(self.U.T, self.label - np.ones(self.archive_size) * self.mu))
        print("mu", self.mu, "sigma: ", self.sigma)

    def likelihood_for_param(self, candidates, args):
        fitness = []
        for ind in candidates:
            f = self.likelihood(ind)[0]
            fitness.append(f)
        return fitness

    def likelihood(self, param):
        theta = param[:self.n_vars]
        exponent = param[self.n_vars:]
        n = self.archive_size
        # build correlation matrix
        R = np.exp(-np.sum(theta * np.power(self.distance, exponent), axis=2))
        R = np.tril(R, -1)  # Return a copy of an array with elements above the k-th diagonal zeroed.
        R = R + R.T + np.eye(n) + np.eye(n) * np.spacing(1)

        try:
            U = np.linalg.cholesky(R)
            U = U.T
        except Exception as e:
            return [np.ones((1, 1)) * 100000, None, 0, 0]
        LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))

        fitness = self.label
        one = np.ones(n)

        mu = (np.dot(one.T, solve(U, solve(U.T, fitness)))) / (np.dot(one.T, solve(U, solve(U.T, one))))
        sigma = (np.dot((fitness - one * mu).T, solve(U, solve(U.T, fitness - one * mu)))) / n
        NegLnLike = .5 * n * np.log(sigma) + .5 * LnDetPsi

        return NegLnLike, R, U, mu, sigma, LnDetPsi

    def predict(self, distance_x):
        theta = self.param[:self.n_vars]
        exponent = self.param[self.n_vars:]

        r = np.array([np.exp(-np.sum((theta * np.power(distance_x, exponent)), axis=1))]).T
        y_hat = self.mu + np.dot(r.T, self.prediction_y_temp)
        sigma_hat = np.abs(self.sigma * (1 - np.dot(r.T, solve(self.U, solve(self.U.T, r)))))
        return y_hat[0], sigma_hat[0][0]   # two float numbers

    def update_distance_incrementally(self, new_distance_row):  # add a new data point into distances.
        self.archive_size += 1
        distance_column = np.zeros((self.archive_size, 1, self.n_vars))
        self.distance = np.append(self.distance, new_distance_row, axis=0)
        self.distance = np.append(self.distance, distance_column, axis=1)

    def update_archive_completely(self, distance, label):
        self.archive_size = len(label)
        self.distance = distance
        self.label = label

    def generate_population(self, random, args):
        '''
        Generates an initial population for any global optimization that occurs in pyKriging
        :param random: A random seed
        :param args: Args from the optimizer, like population size
        :return chromosome: The new generation for our global optimizer to use
        '''
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        """Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.
        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments
        Optional keyword arguments in args:
        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)
        """
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)