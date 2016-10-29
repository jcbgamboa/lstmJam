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

* Python 2.7 / 3.4
* TensorFlow 0.10.0

# Experiments

## MNIST dataset

### Experiment 1

* Number of layers: \[10, 20, 30\]
* Number of epochs: 10/20
* Dropout: NO
* Use tied weights: NO
* Use batch normalization: YES
* Type of weight initialization: Orthogonal
* How-many-folds cross-validation: NO
* Size of state cell: \[100, 200, 300\]

### Experiment 2

* Number of layers: \[10, 20, 30\]
* Number of epochs: 10/20
* Dropout: YES
* Use tied weights: NO
* Use batch normalization: YES
* Type of weight initialization: Orthogonal
* How-many-folds cross-validation: NO
* Size of state cell: \[100, 200, 300\]

### Experiment 3

* Number of layers: \[10, 20, 30\]
* Number of epochs: 10/20
* Dropout: YES
* Use tied weights: YES
* Use batch normalization: YES
* Type of weight initialization: Orthogonal
* How-many-folds cross-validation: NO
* Size of state cell: \[100, 200, 300\]

### Experiment 4

* Number of layers: \[10, 20, 30\]
* Number of epochs: 10/20
* Dropout: NO
* Use tied weights: YES
* Use batch normalization: YES
* Type of weight initialization: Orthogonal
* How-many-folds cross-validation: NO
* Size of state cell: \[100, 200, 300\]

# Results and comparisons

## Results

* Accuracy curve over validation set
* Average accuracy over the entire test set
* Loss curve
* Gradient change over all the parameters
* Images of the batches (Gabor filters?)

## Comparisons

* Compare gradient change with an MLP
* Compare our filters with CNN filters
* Compare others just to see what is the difference (Extra)
* Compare classification results with other models


# TO DO

* Fix visualizations for filters in the for of \[BATCH_SIZE X 28 X 28 X 1\]
    * Remove the present Image Summary
    
* Load a bigger dataset

* Create a comparable CNN to compare the filters in the lower layers and the gradient change over time.

* Allow for any batch size

* Tie weights (problem: size of the bottom-most layer is different)

* Change the Weight initializations
	* Use Normal() [i.e., Gaussian] instead and check results

* Do cross-validation

* Prepare framework for running the experiments automagically

