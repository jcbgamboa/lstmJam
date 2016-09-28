import tensorflow as tf
from network import Network
from tensorflow.examples.tutorials.mnist import input_data
import os


tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("n_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("train", False,
                            "Run a train if this is set to True.")
tf.app.flags.DEFINE_integer("n_epochs", 10,
                            "Number of training iterations.")
tf.app.flags.DEFINE_string("log_dir", "/tmp",
                            "Tensorboard log directory.")
tf.app.flags.DEFINE_string("data_dir", "/tmp",
                            "training data directory.")
FLAGS = tf.app.flags.FLAGS

def create_model(sess, n_input, n_output):

  input = tf.placeholder("float", [FLAGS.batch_size, n_input])
  output = tf.placeholder("float", [FLAGS.batch_size, n_output])
  net = Network(FLAGS.n_layers, FLAGS.size, FLAGS.size, n_input, n_output, input, output)

  # Define loss and optimizer
  with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net.pred, output))

  with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

  # Evaluate model
  with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(net.pred,1), tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Create a summary to monitor cost tensor
  tf.scalar_summary("loss", cost)
  # Create a summary to monitor accuracy tensor
  tf.scalar_summary("accuracy", accuracy)
  # Merge all summaries into a single op
  merged_summary_op = tf.merge_all_summaries()

  saver = tf.train.Saver(tf.all_variables())

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    sess.run(tf.initialize_all_variables())

  return input, output, optimizer, cost, merged_summary_op, accuracy, saver

def train():
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  n_input = 28*28
  n_output = 10
  '''
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  n_input = 28*28
  n_output = 10

  input = tf.placeholder("float", [FLAGS.batch_size, n_input])
  output = tf.placeholder("float", [FLAGS.batch_size, n_output])
  net = Network(FLAGS.n_layers, FLAGS.size, FLAGS.size, n_input, n_output, input, output)

  # Define loss and optimizer
  with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net.pred, output))

  with tf.name_scope('SGD'):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

  # Evaluate model
  with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(net.pred,1), tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initializing the variables
  init = tf.initialize_all_variables()

  # Create a summary to monitor cost tensor
  tf.scalar_summary("loss", cost)
  # Create a summary to monitor accuracy tensor
  tf.scalar_summary("accuracy", accuracy)
  # Merge all summaries into a single op
  merged_summary_op = tf.merge_all_summaries()
  '''
  # Launch the graph
  with tf.Session() as sess:
      #sess.run(init)
      input, output, optimizer, cost, merged_summary_op, accuracy, saver = create_model(sess, n_input, n_output)

      # op to write logs to Tensorboard
      summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

      avg_cost = 0.
      '''
      for i in range(FLAGS.n_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        _, loss, summary = sess.run([optimizer, cost, merged_summary_op],
                                       feed_dict={input: batch_xs, output: batch_ys})
        summary_writer.add_summary(summary, i )
        avg_cost += loss / FLAGS.n_iter
          # Loop over all batches
      '''
      # Training cycle
      for epoch in range(FLAGS.n_epochs):
          avg_cost = 0.
          total_batch = int(mnist.train.num_examples/FLAGS.batch_size)
          # Loop over all batches
          for i in range(total_batch):
              batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
              # Run optimization op (backprop), cost op (to get loss value)
              # and summary nodes
              _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                       feed_dict={input: batch_xs, output: batch_ys})
              # Write logs at every iteration
              summary_writer.add_summary(summary, epoch * total_batch + i)
              # Compute average loss
              avg_cost += c / total_batch

              # Display logs per epoch step
              if (i+1) % FLAGS.steps_per_checkpoint == 0:
                  print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
                  checkpoint_path = os.path.join(FLAGS.train_dir, "transliterate.ckpt")
                  saver.save(sess, checkpoint_path, global_step=epoch*total_batch + i)

      print "Optimization Finished!"

      # Test model
      # Calculate accuracy
      print "Accuracy:", accuracy.eval({input: mnist.test.images, output: mnist.test.labels})

      print "Run the command line:\n" \
            "--> tensorboard --logdir=/tmp/tensorflow_logs " \
            "\nThen open http://0.0.0.0:6006/ into your web browser"

def self_test():
  return

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.train:
    train()

if __name__ == '__main__':
  tf.app.run()
