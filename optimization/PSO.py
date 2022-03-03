# -*- coding: UTF-8 -*-
import numpy as np
import inspyred


class PSO(inspyred.ec.EvolutionaryComputation):
    """Represents a basic particle swarm optimization algorithm.

    This class is built upon the ``EvolutionaryComputation`` class making
    use of an external archive and maintaining the population at the previous
    timestep, rather than a velocity. This approach was outlined in
    (Deb and Padhye, "Development of Efficient Particle Swarm Optimizers by
    Using Concepts from Evolutionary Algorithms", GECCO 2010, pp. 55--62).
    This class assumes that each candidate solution is a ``Sequence`` of
    real values.

    Public Attributes:

    - *topology* -- the neighborhood topology (default topologies.star_topology)

    Optional keyword arguments in ``evolve`` args parameter:

    - *inertia* -- the inertia constant to be used in the particle
      updating (default 0.5)
    - *cognitive_rate* -- the rate at which the particle's current
      position influences its movement (default 2.1)
    - *social_rate* -- the rate at which the particle's neighbors
      influence its movement (default 2.1)

    """

    def __init__(self, random):
        inspyred.ec.EvolutionaryComputation.__init__(self, random)
        self.topology = inspyred.swarm.topologies.star_topology
        self._previous_population = []
        self.selector = self._swarm_selector
        self.replacer = self._swarm_replacer
        self.variator = self._swarm_variator
        self.archiver = self._swarm_archiver

    def _swarm_archiver(self, random, population, archive, args):
        if len(archive) == 0:
            return population[:]
        else:
            new_archive = []
            for i, (p, a) in enumerate(zip(population[:], archive[:])):
                if p < a:
                    new_archive.append(a)
                else:
                    new_archive.append(p)
            return new_archive

    def _swarm_variator(self, random, candidates, args):
        inertia = args.setdefault('inertia', 0.5)
        cognitive_rate = args.setdefault('cognitive_rate', 1.5)
        social_rate = args.setdefault('social_rate', 1.5)
        if len(self.archive) == 0:
            self.archive = self.population[:]
        if len(self._previous_population) == 0:
            self._previous_population = self.population[:]
        neighbors = self.topology(self._random, self.archive, args)
        offspring = []
        for x, xprev, pbest, hood in zip(self.population,
                                         self._previous_population,
                                         self.archive,
                                         neighbors):
            nbest = max(hood)
            particle = []
            for xi, xpi, pbi, nbi in zip(x.candidate, xprev.candidate,
                                         pbest.candidate, nbest.candidate):
                value = (xi + inertia * (xi - xpi) +
                         cognitive_rate * random.random() * (pbi - xi) +
                         social_rate * random.random() * (nbi - xi))
                particle.append(value)
            particle = self.bounder(particle, args)
            offspring.append(particle)
        return offspring

    def _swarm_selector(self, random, population, args):
        return population

    def _swarm_replacer(self, random, population, parents, offspring, args):
        self._previous_population = population[:]
        return offspring

def generate_population(random, args):
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

def no_improvement_termination(population, num_generations, num_evaluations, args):
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
    max_generations = args.setdefault('max_generations', 30)
    previous_best = args.setdefault('previous_best', None)
    max_evaluations = args.setdefault('max_evaluations', 30000)

    current_best = np.around(min(population).fitness, decimals=4)
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