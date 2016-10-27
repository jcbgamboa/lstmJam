import time
import uuid
import os
import tensorflow as tf
from lstm import BNLSTMCell, orthogonal_initializer
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("n_layers", 10, "Number of layers in the model.")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 500,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("train", True,
                            "Run a train if this is set to True.")
tf.app.flags.DEFINE_integer("n_itr", 100000,
                            "Number of training iterations.")
tf.app.flags.DEFINE_string("log_dir", "/tmp",
                            "Tensorboard log directory.")
tf.app.flags.DEFINE_string("data_dir", "/tmp",
                            "training data directory.")
FLAGS = tf.app.flags.FLAGS

def data_prep():
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  return mnist

def create_model(input_size,output_size,batch_size=128,hidden_size=128,n_layers=10):
  #batch_size = 128
  #hidden_size = 128
  #n_layers = 10

  x = tf.placeholder(tf.float32, [None, input_size])
  training = tf.placeholder(tf.bool)

  #c, h
  initialState = (tf.random_normal([batch_size, hidden_size], stddev=0.1),
      tf.random_normal([batch_size, hidden_size], stddev=0.1))

  list_layers = []
  id = 1
  cell_1 = BNLSTMCell(hidden_size, training=training)
  new_h, new_state = cell_1(x, initialState, id, first=True)

  layers = [cell_1]
  prev_cell = cell_1

  for l in range(1, (n_layers-1)):
    id += 1
    next_cell = BNLSTMCell(hidden_size, training=training)
    next_new_h, next_new_state = next_cell(prev_cell.new_h, prev_cell.state, id, first=True)
    layers.append(next_cell)
    prev_cell = layers[-1]

  outputs, state = layers, prev_cell.state

  _, final_hidden = state

  W = tf.get_variable('W', [hidden_size, output_size], initializer=orthogonal_initializer())
  b = tf.get_variable('b', [output_size])

  y = tf.nn.softmax(tf.matmul(final_hidden, W) + b)

  y_ = tf.placeholder(tf.float32, [None, output_size])

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  gvs = optimizer.compute_gradients(cross_entropy)
  capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
  train_step = optimizer.apply_gradients(capped_gvs)

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Summaries
  tf.scalar_summary("accuracy", accuracy)
  tf.scalar_summary("xe_loss", cross_entropy)
  for (grad, var), (capped_grad, _) in zip(gvs, capped_gvs):
      if grad is not None:
          tf.histogram_summary('grad/{}'.format(var.name), capped_grad)
          tf.histogram_summary('capped_fraction/{}'.format(var.name),
              tf.nn.zero_fraction(grad - capped_grad))
          tf.histogram_summary('weight/{}'.format(var.name), var)

  for k,layer in enumerate(outputs):
    w_i, w_j, w_f, w_o = tf.split(1, 4, layer.W_xh)

    w_i = tf.reshape(w_i,(1,w_i.get_shape().as_list()[0],w_i.get_shape().as_list()[1],1))
    w_j = tf.reshape(w_j,(1,w_j.get_shape().as_list()[0],w_j.get_shape().as_list()[1],1))
    w_f = tf.reshape(w_f,(1,w_f.get_shape().as_list()[0],w_f.get_shape().as_list()[1],1))
    w_o = tf.reshape(w_o,(1,w_o.get_shape().as_list()[0],w_o.get_shape().as_list()[1],1))

    tf.image_summary("layer_{}_w_i".format(k), w_i)
    tf.image_summary("layer_{}_w_j".format(k), w_j)
    tf.image_summary("layer_{}_w_f".format(k), w_f)
    tf.image_summary("layer_{}_w_o".format(k), w_o)

  merged = tf.merge_all_summaries()
  return merged, train_step, cross_entropy, x, y_, training, accuracy

def load_model(saver,sess,chkpnts_dir):
  ckpt = tf.train.get_checkpoint_state(chkpnts_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("Loading previously trained model: {}".format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("Training with fresh parameters")
    sess.run(tf.initialize_all_variables())

def train():
  mnist = data_prep()

  merged, train_step, cross_entropy, x, y_, training, accuracy = create_model(784,
                                                                    10,
                                                                    FLAGS.batch_size,
                                                                    FLAGS.size,
                                                                    FLAGS.n_layers)

  saver = tf.train.Saver(tf.all_variables())
  sess = tf.Session()
  checkpoints_folder = './chkpnts/'
  if not os.path.exists(checkpoints_folder):
    os.makedirs(checkpoints_folder)
  load_model(saver, sess, "chkpnts/")
  #init = tf.initialize_all_variables()
  #sess.run(init)

  logdir = 'logs/' + str(uuid.uuid4())
  os.makedirs(logdir)
  print('logging to ' + logdir)
  writer = tf.train.SummaryWriter(logdir, sess.graph)

  current_time = time.time()
  print("Using population statistics (training: False) at test time gives worse results than batch statistics")

  for i in range(FLAGS.n_itr):
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys, training: True})
      step_time = time.time() - current_time
      current_time = time.time()
      if i % FLAGS.steps_per_checkpoint == 0:
        batch_xs, batch_ys = mnist.validation.next_batch(FLAGS.batch_size)
        summary_str = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, training: False})
        writer.add_summary(summary_str, i)
        checkpoint_path = os.path.join("chkpnts/", "lstmjam.ckpt")
        saver.save(sess, checkpoint_path, global_step=i)
        print(loss, step_time, i)
        avg_acc = 0.0
        for test_itr in range(70):
          test_data, test_label = mnist.test.next_batch(FLAGS.batch_size)
          acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_label, training: False})
          avg_acc += acc
          #test_label = mnist.test.labels[:FLAGS.batch_size]
        print("Testing Accuracy:" + str(avg_acc/70))

def test():
  mnist = data_prep()

  merged, train_step, cross_entropy, x, y_, training, accuracy = create_model(784,
                                                                    10,
                                                                    FLAGS.batch_size,
                                                                    FLAGS.size,
                                                                    FLAGS.n_layers)

  saver = tf.train.Saver(tf.all_variables())
  sess = tf.Session()
  load_model(saver, sess, "chkpnts/")
  test_data = mnist.test.images
  test_label = mnist.test.labels
  print("Testing Accuracy:" + str(sess.run(accuracy, feed_dict={x: test_data, y_: test_label, training: False})))

def main(_):
  if FLAGS.self_test:
    pass
  elif FLAGS.train:
    train()

if __name__ == '__main__':
  tf.app.run()
