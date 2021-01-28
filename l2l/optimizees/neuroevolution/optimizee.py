import numpy as np
import os
import pandas as pd
import subprocess
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

NeuroEvolutionOPtimizeeParameters = namedtuple(
    'NeuroEvoOptimizeeParameters', ['seed', 'test_size', 'path'])


class NeuroEvolutionOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.rng = np.random.default_rng()
        self.path = parameters.path


def create_individual(self):
    """
    Creates and returns the individual

    Will launch netlogo and create a network instance to get the parameters.
    Parameters are read in via a csv file and returned as a dictionary.
    """
    csv_path = './'
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
    # run netlogo
    run_cmd = 'netlogo-headless.bat --model /home/research/l2l-netlogo/agent-individual.nlogo --experiment experiment1 --table /home/research/l2l-netlogo/table1.csv'
    subprocess.run(run_cmd)
    # wait ?

    csv_path = os.pash.join(
        self.path, 'individual{}.csv'.format(traj.individual.ind_idx))
    csv = pd.read_csv(csv_path)
    fitness = csv.iloc[3]
    return fitness


def bounding_func(self, individual):
    return individual
