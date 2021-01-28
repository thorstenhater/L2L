import numpy as np
from datetime import datetime
from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.optimizees.neuroevolution import NeuroEvolutionOptimizee, NeuroEvolutionOptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer


def run_experiment():
    experiment = Experiment(
        root_dir_path='/p/scratch/structuretofunction/l2l_alper/')
    jube_params = {"exec": "srun -n 1 -c 8 --exclusive python"}
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name="NeuroEvo_ES_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))

    # Optimizee params
    optimizee_parameters = NeuroEvolutionOptimizee(
        path=experiment.root_dir_path)
    optimizee = NeuroEvolutionOptimizee(traj, optimizee_parameters)

    optimizer_seed = 1234
    optimizer_parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        noise_std=0.1,
        mirrored_sampling_enabled=True,
        fitness_shaping_enabled=True,
        pop_size=20,
        n_iteration=2000,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(1.,),
        parameters=optimizer_parameters,
        optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
