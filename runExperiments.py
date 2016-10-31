import tensorflow as tf
import trainBatchNorm

import inspect, sys

FLAGS = tf.app.flags.FLAGS

def set_flags(FLAGS, n_layers, epochs, state_cells, dropout, tie_weights):
	FLAGS.n_layers = n_layers
	FLAGS.n_epochs = epochs
	FLAGS.size = state_cells
	FLAGS.dropout = dropout
	FLAGS.tie_weights = tie_weights

################################################################### EXPERIMENT 1
def exp1_10layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, False)

def exp1_20layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, False)

def exp1_30layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, False)

def exp1_10layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, False)

def exp1_20layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, False)

def exp1_30layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, False)

def exp1_10layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, False)

def exp1_20layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, False)

def exp1_30layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, False)

def exp1_10layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, False)

def exp1_20layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, False)

def exp1_30layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, False)

################################################################### EXPERIMENT 2
def exp2_10layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, False)

def exp2_20layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, False)

def exp2_30layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, False)

def exp2_10layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, False)

def exp2_20layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, False)

def exp2_30layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, False)

def exp2_10layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, False)

def exp2_20layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, False)

def exp2_30layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, False)

def exp2_10layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, False)

def exp2_20layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, False)

def exp2_30layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, False)


################################################################### EXPERIMENT 3
def exp3_10layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, True)

def exp3_20layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, True)

def exp3_30layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, True)

def exp3_10layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, True)

def exp3_20layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, True)

def exp3_30layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, True)

def exp3_10layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, True)

def exp3_20layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, True)

def exp3_30layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, True)

def exp3_10layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 0.5, True)

def exp3_20layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 0.5, True)

def exp3_30layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 0.5, True)


################################################################### EXPERIMENT 4

def exp4_10layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, True)

def exp4_20layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, True)

def exp4_30layers_10epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, True)

def exp4_10layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, True)

def exp4_20layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, True)

def exp4_30layers_20epochs_100statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, True)

def exp4_10layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, True)

def exp4_20layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, True)

def exp4_30layers_10epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, True)

def exp4_10layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 10, 10, 100, 1, True)

def exp4_20layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 20, 10, 100, 1, True)

def exp4_30layers_20epochs_200statecells(FLAGS):
	set_flags(FLAGS, 30, 10, 100, 1, True)


################################################################## MAIN FUNCTION

def get_experiments(regex):
	experiments = [obj for name,obj in inspect.getmembers(
						sys.modules[__name__])
				if (inspect.isfunction(obj) and
						regex in name)]

	return experiments

tf.app.flags.DEFINE_string("run_experiments", '',
			"What experiments to runi.")

def main(_):
	experiments = get_experiments(FLAGS.run_experiments)

	for e in experiments:
		e(FLAGS)
		print(("n_layers: {}, n_epochs: {}, size: {}, " +
			"dropout: {}, tie: {}").format(
				FLAGS.n_layers, FLAGS.n_epochs, FLAGS.size,
				FLAGS.dropout, FLAGS.tie_weights))
		trainBatchNorm.train(False)

if __name__ == '__main__':
	tf.app.run()

