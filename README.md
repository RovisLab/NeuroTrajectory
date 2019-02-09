# NeuroTrajectory
##A Neuroevolutionary Approach to Local State Trajectory Learning in Autonomous Driving

This repository accompanies the paper *NeuroTrajectory: A Neuroevolutionary Approach to Local State Trajectory Learning in Autonomous Driving*.

Current autonomous vehicles are controlled either based on a decoupled perception-planning-action sequence of operations, either based on End2End or Deep Reinforcement Learning (DRL) approaches. Deep learning solutions are subject to several limitations (e.g. they only compute the best driving action for the next upcoming sampling time, in a discrete form: turn left, turn right, accelerate, break). The learning method uses a single-objective loss function and the backpropagation algorithm for learning a direct mapping of the input data to discrete steering commands. To address these issues, we introduce *NeuroTrajectory*, which is a multi-objective neuroevolutionary approach to local trajectory learning for autonomous driving, where the desired trajectory of the ego-vehicle is estimated over a finite prediction horizon by a *perception-planning* deep neural network. We propose an approach which uses genetic algorithms for training a population of deep neural networks, where each network individual is evaluated based on a multi-objective fitness vector, with the purpose of establishing a so-called *Pareto front* of optimal deep neural networks. The performance of an individual is given by a fitness vector composed of three elements. Each element describes the vehicle's travel path, lateral velocity and longitudinal speed, respectively. The same network structure can be trained on synthetic, as well as on real-world data sequences. We have benchmarked our system against a baseline Dynamic Windows Approach (DWA), as well as against an End2End learning method.

![Alt text](images/pareto_optimization_problem.png?raw=true)

## Installation

## Training a model
