![alt tag](https://sigvoiced.files.wordpress.com/2016/07/tlstm_full.png)
-----------------

This project has the goal of experimenting with an unfolded version
of LSTMs.

It is easy to see that a LSTM cell, when unfolded into several time-
steps, as is the case when, e.g., Backpropagation Through Time is
applied, composes a very deep Neural Networks (in the case of BPTT,
with as many layers as is the number of time-steps) where the weights
connecting each of the layers are "tied", i.e., are the same for all
layers.

**Hypothesis:** The gradients of a deep Neural Network following the
same architecture of the LSTM unfolded through time (even those of the
bottom layers) are efficiently trainable with Backpropagation, and
won't be affected by the "vanishing gradient" problem. This is the
case even when the weights are not "tied".

Our belief is that the gradients will propagate through the "state
cell" (or, actually, its equivalent) to the lower layers.

If this _Hypothesis_ is true, another interesting question is what
exactly will the Network learn when trained to, e.g., classify the
handwritten digit images of MNIST. Will the bottom layer somehow learn
filters similar to Gabor filters, just like CNNs, Auto-encoders, RBMs
or Sparsity models?

# Dependencies

We have tested the code in the following two environments:

* Python 3.4
* TensorFlow 0.9.0

[get the info about the other environment]

# Experiments

[this is a simple format for describing the experiments]

## MNIST dataset

### Experiment 1

* Number of layers:
* Use tied weights:
* Use batch normalization:
* Type of weight initialization:
* How-many-folds cross-validation:
* Size of state cell:
* Size of output cell: (can this be different from the size of the
	state cell?)

### Experiment 2


# TO DO

* Prepare framework for running the experiments automagically

* Change the Weight initializations
	* Use Normal() [i.e., Gaussian] instead and check results

* Do cross-validation

* Try Drop-Out (we believe this network is likely to overfit,
	especially when "tied weights" are not used)

* Tie weights (problem: size of the bottom-most layer is different)

