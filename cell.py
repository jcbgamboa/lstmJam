import tensorflow as tf
import numpy as np

class Cell(object):
  def __init__(self, state, h_activation, input=None, first=False):

    if first and input is not None:
      h_activation = tf.concat(0,[tf.transpose(input), h_activation])

    blah = tf.Variable(tf.constant(0.1,shape=[256]))

    self.wf,self.bf,self.wi,self.bi,self.wc,self.bc,self.wo,self.bo = \
                    self.initialize_parameters(state.get_shape().as_list()[0],
                                        h_activation.get_shape().as_list()[0],
                                        first)

    f = tf.sigmoid(tf.matmul(self.wf, h_activation) + self.bf)
    i = tf.sigmoid(tf.matmul(self.wi, h_activation) + self.bi)
    c = tf.tanh(tf.matmul(self.wc, h_activation) + self.bc)
    o = tf.sigmoid(tf.matmul(self.wo, h_activation) + self.bo)

    self.state_next = tf.add(tf.mul(f, state),tf.mul(i, c))
    self.h_next = tf.mul(tf.tanh(self.state_next), o)
    return

  def initialize_parameters(self,n_s,n_h,first):
    #print("n_s: {a}; n_h: {b}\n".format(a=n_s, b=n_h))
    wf = tf.Variable(tf.truncated_normal([n_s,n_h],stddev=0.1))
    bf = tf.Variable(tf.constant(0.1,shape=[n_s,1]))
    wi = tf.Variable(tf.truncated_normal([n_s,n_h],stddev=0.1))
    bi = tf.Variable(tf.constant(0.1,shape=[n_s,1]))
    wc = tf.Variable(tf.truncated_normal([n_s,n_h],stddev=0.1))
    bc = tf.Variable(tf.constant(0.1,shape=[n_s,1]))
    wo = tf.Variable(tf.truncated_normal([n_s,n_h],stddev=0.1))
    bo = tf.Variable(tf.constant(0.1,shape=[n_s,1]))
    return wf, bf, wi, bi, wc, bc, wo, bo

