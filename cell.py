import tensorflow as tf
import numpy as np


class Cell(object):
  def __init__(self, state, h_activation,
                input = None, first = False, prevCell = None):

    if first and input is not None:
      # Concatenates the vector of inputs with the vector of activations
      h_activation = tf.concat(0, [tf.transpose(input), h_activation])

    if first or prevCell is None:
      # Generates the gating matrices and biases
      print("Will initialize the first parameters...")
      self.wf, self.wi, self.wc, self.wo, \
      self.bf, self.bi, self.bc, self.bo = \
                    self.initialize_parameters(state.get_shape().as_list()[0],
                                        h_activation.get_shape().as_list()[0],
                                        first)
    else: #if not first and prevCell is not None:
      print("Will reuse awkeufahweufhkawe...")
      # TODO: Maybe create a class for these
      self.wf, self.wi, self.wc, self.wo, \
      self.bf, self.bi, self.bc, self.bo = \
                    prevCell.wf, prevCell.wi, prevCell.wc, prevCell.wo, \
                    prevCell.bf, prevCell.bi, prevCell.bc, prevCell.bo,

    f = tf.sigmoid(tf.matmul(self.wf, h_activation) + self.bf)
    i = tf.sigmoid(tf.matmul(self.wi, h_activation) + self.bi)
    c = tf.tanh(tf.matmul(self.wc, h_activation) + self.bc)
    o = tf.sigmoid(tf.matmul(self.wo, h_activation) + self.bo)

    self.state_next = tf.add(tf.mul(f, state), tf.mul(i, c))
    self.h_next = tf.mul(tf.tanh(self.state_next), o)
    return

  def initialize_parameters(self, n_s, n_h, first):
    """ Obviously, this function can be simplified.
        It is this way only to aid debugging. """

    wf = tf.Variable(tf.truncated_normal([n_s, n_h], stddev = 0.1))
    wi = tf.Variable(tf.truncated_normal([n_s, n_h], stddev = 0.1))
    wc = tf.Variable(tf.truncated_normal([n_s, n_h], stddev = 0.1))
    wo = tf.Variable(tf.truncated_normal([n_s, n_h], stddev = 0.1))

    bf = tf.Variable(tf.constant(0.1, shape = [n_s, 1]))
    bi = tf.Variable(tf.constant(0.1, shape = [n_s, 1]))
    bc = tf.Variable(tf.constant(0.1, shape = [n_s, 1]))
    bo = tf.Variable(tf.constant(0.1, shape = [n_s, 1]))

    return wf, wi, wc, wo, bf, bi, bc, bo

