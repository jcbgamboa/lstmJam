import tensorflow as tf
import numpy as np

class Cell(object):
  def __init__(self, state, h_activation, input=None, first=False):

    if first and input is not None:
      h_activation = tf.concat(1,[tf.transpose(input), h_activation])

    self.wf,self.bf,self.wi,self.bi,self.wc,self.bc,self.wo,self.bo = \
                              self.initialize_parameters(state.shape[1],
                                                    h_activation.shape[1],
                                                    first)

    f = tf.sigmoid(tf.matmul(self.wf, h_activation) + self.bf)
    i = tf.sigmoid(tf.matmul(self.wi, h_activation) + self.bi)
    c = tf.tanh(tf.matmul(self.wc, h_activation) + self.bc)
    o = tf.sigmoid(tf.matmul(self.wo, h_activation) + self.bo)

    self.state_next = tf.add(tf.mul(f, state),tf.mul(i, c))
    self.h_next = tf.mul(tf.tanh(self.state_next), o)

  def initialize_parameters(self,n_s,n_h,first):
    return tf.Variable(tf.truncated_normal(n_s,n_h,stddev=0.1)),\
           tf.Variable(tf.constant(0.1,shape=n_s)),\
           tf.Variable(tf.truncated_normal(n_s,n_h,stddev=0.1)),\
           tf.Variable(tf.constant(0.1,shape=n_s)),\
           tf.Variable(tf.truncated_normal(n_s,n_h,stddev=0.1)),\
           tf.Variable(tf.constant(0.1,shape=n_s)),\
           tf.Variable(tf.truncated_normal(n_s,n_h,stddev=0.1)),\
           tf.Variable(tf.constant(0.1,shape=n_s))

