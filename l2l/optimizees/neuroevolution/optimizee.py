import numpy as np
import os
import pandas as pd
import shutil
import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee

NeuroEvolutionOptimizeeParameters = namedtuple(
    'NeuroEvoOptimizeeParameters', ['path', 'seed'])


class NeuroEvolutionOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.param_path = parameters.path
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation
        seed = parameters.seed * self.ind_idx + self.generation
        self.rng = np.random.default_rng(seed)
        self.dir_path = ''

    def create_individual(self):
        """
        Creates and returns the individual

        Creates the parameters for netlogo.
        The parameter are `weights`, `plasticity` and `delays`.
        """
        # TODO the creation of the parameters should be more variable
        #  e.g. as parameters or as a config file
        # create random weights
        weights = self.rng.uniform(-15, 15, 78)
        # create plasticity {0,1} for on/off
        plasticity = self.rng.integers(2, size=78)
        # create delays
        delays = self.rng.integers(low=1, high=7, size=78)
        # create individual
        individual = {
            'weights': weights,
            'plasticity': plasticity,
            'delays': delays
        }
        return individual

    def simulate(self, traj):
        """
        Simulate a run and return a fitness

        A directory `individualN` and a csv file `individualN` with parameters
        to optimize will be saved.
        Invokes a run of netlogo, reads in a file outputted (`resultN`) by
        netlogo with the fitness inside.
        """
        weights = traj.individual.weights
        plasticity = traj.individual.plasticity
        delays = traj.individual.delays
        # create directory individualN
        self.dir_path = os.path.join(self.param_path,
                                     'individual{}'.format(self.ind_idx))
        os.mkdir(self.dir_path)
        individual = {
            'weights': weights,
            'plasticity': plasticity,
            'delays': delays
        }
        # create the csv file and save it in the created directory
        df = pd.DataFrame(individual)
        df = df.T
        df.to_csv(os.path.join(self.dir_path, 'individual_config.csv'.format(self.ind_idx)),
                  header=False, index=False)
        # copy model to the created directory
        # TODO adapt next line or make it a parameter for the bin file
        #  or config file
        model_path = ''
        shutil.copyfile(model_path, os.path.join(self.dir_path, 'model.nlogo'))
        # call netlogo
        subdir_path = os.path.join(self.dir_path, 'model.nlogo')
        os.system(
            'bash /opt/netlogo/netlogo-headless.sh --model {} --experiment experiment1 --table table1.csv'.format(subdir_path))
        file_path = os.path.join(self.dir_path, "individual_result.csv")
        while not os.path.isfile(file_path):
            time.sleep(5)
        # Read the results file after the netlogo run
        csv = pd.read_csv(file_path)
        fitness = csv.iloc[-1]
        # remove directory 
        shutil.rmtree(self.dir_path)
        return fitness

    def bounding_func(self, individual):
        return individual
