# NeuroTrajectory
## A Neuroevolutionary Approach to Local State Trajectory Learning for Autonomous Vehicles

This repository accompanies the paper *NeuroTrajectory: A Neuroevolutionary Approach to Local State Trajectory Learning in Autonomous Driving*.

Autonomous vehicles are controlled today either based on sequences of decoupled perception-planning-action operations, either based on End2End or Deep Reinforcement Learning (DRL) systems. Deep learning solutions are subject to several limitations (e.g. they only compute the best driving action for the next upcoming sampling time, in a discrete form: turn left, turn right, accelerate, break). The learning method uses a single-objective loss function and the backpropagation algorithm for learning a direct mapping of the input data to discrete steering commands. To address these issues, we introduce *NeuroTrajectory*, which is a multi-objective neuroevolutionary approach to local trajectory learning for autonomous driving, where the desired trajectory of the ego-vehicle is estimated over a finite prediction horizon by a *perception-planning* deep neural network. We propose an approach which uses genetic algorithms for training a population of deep neural networks, where each network individual is evaluated based on a multi-objective fitness vector, with the purpose of establishing a so-called *Pareto front* of optimal deep neural networks. The performance of an individual is given by a fitness vector composed of three elements. Each element describes the vehicle's travel path, lateral velocity and longitudinal speed, respectively. The same network structure can be trained on synthetic, as well as on real-world data sequences. We have benchmarked our system against a baseline Dynamic Windows Approach (DWA), as well as against an End2End learning method.

![Alt text](images/pareto_optimization.png?raw=true)

## Installation

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Clone the repository:
```bash
$ git clone https://github.com/RovisLab/NeuroTrajectory.git
```

The packages needed for install can be found inside requirements.txt: 

```
pip install -r requirements.txt
```

### Running the code

The script which runs the main function is

```
main.py
```

### Training a model

data_types.py contains the configuration parameters for the 3 possible types of networks: DGN, LSTM and Conv3D

Modify DATA_SET_INFO/data_set_path to point to the training data (split in training, validation and testing).

Already prepared data and stored in numpy format will be provided as .zip.

At the end of each training, plots and .csv files are generated in ./train/(date_time)/

## Built with

* [SciKit Learn][https://scikit-learn.org/stable/] - Machine Learning with Python.
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.