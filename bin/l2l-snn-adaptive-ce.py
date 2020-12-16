import numpy as np
from l2l.utils.experiment import Experiment
from l2l.optimizees.snn.optimizee_ce import OptimizeeCE, \
    OptimizeeCEParameters
from l2l.optimizers.crossentropy import CrossEntropyOptimizerSNN, \
    CrossEntropyParametersSNN
from l2l.optimizers.crossentropy.distribution import NoisyGaussian


def run_experiment():
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(jube_parameter={}, name="L2L-ENKF")

    # Optimizee params
    optimizee_parameters = OptimizeeCEParameters(
        path=experiment.root_dir_path,
        record_spiking_firingrate=True,
        save_plot=False)
    # Inner-loop simulator
    optimizee = OptimizeeCE(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    pop_size = 2
    optimizer_parameters = CrossEntropyParametersSNN(rho=0.9,
                                                     pop_size=pop_size,
                                                     smoothing=0.1,
                                                     temp_decay=0.1,
                                                     n_iteration=3,
                                                     n_batches=2,
                                                     distribution=NoisyGaussian(
                                                         noise_magnitude=1.,
                                                         noise_decay=0.99),
                                                     stop_criterion=np.inf,
                                                     seed=optimizer_seed)

    optimizer = CrossEntropyOptimizerSNN(traj,
                                         optimizee_prepare=optimizee.connect_network,
                                         optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1.,),
                                         parameters=optimizer_parameters,
                                         optimizee_bounding_func=None)
    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
