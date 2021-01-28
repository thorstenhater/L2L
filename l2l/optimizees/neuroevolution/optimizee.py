import numpy as np
import os
import pandas as pd
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

NeuroEvolutionOptimizeeParameters = namedtuple(
    'NeuroEvoOptimizeeParameters', ['path'])


class NeuroEvolutionOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.rng = np.random.default_rng()
        self.path = parameters.path
        self.ind_idx=traj.individual.ind_idx

    def create_individual(self):
        """
        Creates and returns the individual

        Will launch netlogo and create a network instance to get the parameters.
        Parameters are read in via a csv file and returned as a dictionary.
        """
        csv_path = os.path.join('.', 'individual_config_{}.csv'.format(self.ind_idx))
        print(csv_path)
        csv = pd.read_csv(csv_path)
        individual = {
            'weights': csv.iloc[0],
            'plasticity': csv.iloc[1],
            'delay': csv.iloc[2]
        }
        return individual

    def simulate(self, traj):
        """
        Simulate a run and return a fitness

        Invokes a run of netlogo, reads in a file outputted by netlogo with
        parameters inside to optimize.
        """
        csv_path = os.path.join(
            self.path, 'individual{}.csv'.format(traj.individual.ind_idx))
        csv = pd.read_csv(csv_path)
        fitness = csv.iloc[3]
        return fitness

    def bounding_func(self, individual):
        return individual
