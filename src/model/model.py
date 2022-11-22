# import forestci as fci
import random
# import forestci as fci
import warnings
import tensorflow as tf

import six
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN

from src.data_handeling.Preprocess import *
from tensorflow.keras.layers import TimeDistributed, RepeatVector
# from src.model.model import *

#my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
#tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)

# Neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from keras.regularizers import l1

# Wrapper to make neural network compitable with StackingRegressor

rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
warnings.filterwarnings("ignore")
import yaml

optimisers = ['Adam']
im = 0
from mlinsights.mlmodel import IntervalRegressor

precision = []
recall = []
Accuracy = []
F1 = []
force_gc = True
import logging
import sys

with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
    # Load the data
    config_path = experiment_config["experiment_settings"]["config_path"]
    lookback = experiment_config['data_parameters']['look_back']


def generate_rnn(hidden_layers, input_shape):
    """
    Generates a RNN using an array of hidden layers including the number of neurons for each layer
    :param hidden_layers:
    :return:
    """

    # Create and fit the RNN
    model = Sequential()
    # model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # Add input layer
    model.add(Dense(500, input_shape=(input_shape, experiment_config['data_parameters']['look_back'])))

    # Add hidden layers
    for i in range(len(hidden_layers)):

        if i == 0:
            neurons_layer = hidden_layers[i]
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters'][
                'max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            #         model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            #        model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        #       model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif i == 1:
            neurons_layer = int(hidden_layers[0] / 4)
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters'][
                'max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking
            #            pdb.set_trace()

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        else:
            neurons_layer = int(hidden_layers[1] / 4)
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters'][
                'max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))

    # Add output layer
    model.add(Dense(input_shape))
    # Compile the RNN
    model.compile(loss='mean_squared_error', optimizer=optimisers[0])
    return model


def generate_rf(estimators):
    """
    Generates a Random Forest with the number of estimators to use
    :param estimators:
    :return:
    """
    # Create and fit the RF
    model = RandomForestRegressor(n_estimators=estimators, criterion='mse', max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=4, max_features='auto', max_leaf_nodes=None,
                                  bootstrap=2,
                                  oob_score=False, n_jobs=4, random_state=None, verbose=0)

    return model


def autoencodermodel(input_shape):
    """
    Generates a autoencoder model
    :param input_shape:
    :return:
    """
    # Create and fit the RNN
    m = Sequential()

    m.add(Dense(512, input_shape=(input_shape,), name='encoder_1'))
    m.add(Dense(128, activation='elu', name='encoder_2'))
    m.add(Dense(5, activation='linear', name="encoder_3"))
    m.add(Dense(128, activation='elu', name='decoder_1'))
    m.add(Dense(512, activation='elu', name='decoder_2'))
    m.add(Dense(input_shape, activation='sigmoid', name='output'))

    m.compile(loss='mean_squared_error', optimizer='adam')
    return m
def lstm_autoencoder(input_shape):
    model = Sequential()
    model.add(LSTM(256, activation='relu', batch_input_shape=(None, 200,38), return_sequences=True,
              ))
    model.add(LSTM(128, activation='relu', return_sequences=False, name='lstm2'))
    model.add(RepeatVector(200))
    model.add(LSTM(128, activation='relu', return_sequences=True, name='lstm3'))
    model.add(LSTM(256, activation='relu', return_sequences=True, name='lstm4'))
    model.add(TimeDistributed(Dense(38)))
    model.compile(loss='mae', optimizer='adam')
    return model


def generate_LinearRegression():
    """
    Generates a Linear Regression"
    """
    model = IntervalRegressor(LinearRegression(), n_estimators=50, alpha=0.05)
    return model


def create_neural_network(input_shape, depth=5, batch_mod=2, num_neurons=20, drop_rate=0.1, learn_rate=.01,
                          r1_weight=0.02,
                          r2_weight=0.02):
    '''A neural network architecture built using keras functional API'''
    act_reg = l1(r2_weight)
    kern_reg = l1(r1_weight)

    inputs = Input(shape=(input_shape,))
    batch1 = BatchNormalization()(inputs)
    hidden1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(batch1)
    dropout1 = Dropout(drop_rate)(hidden1)
    # lstm1=LSTM(num_neurons,  dropout=drop_rate, recurrent_dropout=drop_rate)(dropout1)
    batch2 = BatchNormalization()(dropout1)

    hidden2 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
        dropout1)

    skip_list = [batch1]
    last_layer_in_loop = hidden2
    # pdb.set_trace()

    for i in range(depth):
        added_layer = concatenate(skip_list + [last_layer_in_loop])
        skip_list.append(added_layer)
        b1 = None
        # Apply batch only on every i % N layers
        if i % batch_mod == 2:
            b1 = BatchNormalization()(added_layer)
        else:
            b1 = added_layer

        h1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(b1)
        d1 = Dropout(drop_rate)(h1)
        h2 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d1)
        d2 = Dropout(drop_rate)(h2)
        h3 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d2)
        d3 = Dropout(drop_rate)(h3)
        h4 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d3)
        last_layer_in_loop = h4
        c1 = concatenate(skip_list + [last_layer_in_loop])
        output = Dense(1, activation='sigmoid')(c1)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam()
    optimizer.learning_rate = learn_rate

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    return model




class LSTM_Var_Autoencoder(object):

    def __init__(self, intermediate_dim=None, z_dim=None, n_dim=None, kulback_coef=0.1,
                 stateful=False):
        """
        Args:
        intermediate_dim : LSTM cells dimension.
        z_dim : dimension of latent space.
        n_dim : dimension of input data.
        statefull : if true, keep cell state through batches.
        """

        if not intermediate_dim or not z_dim or not n_dim:
            raise ValueError("You should set intermediate_dim, z_dim"
                             "(latent space) dimension and your input"
                             "third dimension, n_dim."
                             " \n            ")

        tf.reset_default_graph()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.intermediate_dim = intermediate_dim
        self.stateful = stateful
        self.input = tf.placeholder(tf.float32, shape=[None, None, self.n_dim])
        self.batch_size = tf.placeholder(tf.int64)
        self.kulback_coef = kulback_coef
        # tf.data api
        dataset = tf.data.Dataset.from_tensor_slices(self.input).repeat() \
            .batch(self.batch_size)
        self.batch_ = tf.placeholder(tf.int32, shape=[])
        self.ite = dataset.make_initializable_iterator()
        self.x = self.ite.get_next()
        self.repeat = tf.placeholder(tf.int32)

        def gauss_sampling(mean, sigma):
            with tf.name_scope("sample_gaussian"):
                eps = tf.random_normal(tf.shape(sigma), 0, 1, dtype=tf.float32)
                # It should be log(sigma / 2), but this empirically converges"
                # much better for an unknown reason"
                z = tf.add(mean, tf.exp(0.5 * sigma) * eps)
                return z

        # (with few modifications) from https://stackoverflow.com/questions

        def get_state_variables(batch_size, cell):
            # For each layer, get the initial state and make a variable out of it
            # to enable updating its value.
            state_variables = []
            for state_c, state_h in cell.zero_state(batch_size, tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    (state_c), (state_h)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial
            # state
            return tuple(state_variables)

        # Add an operation to update the train states with the last state
        # tensors
        def get_state_update_op(state_variables, new_states):
            update_ops = []
            for state_variable, new_state in zip(state_variables, new_states):
                update_ops.extend([state_variable[0] == new_state[0],
                                   state_variable[1] == new_state[1]])
            return tf.tuple(update_ops)

        # Return an operation to set each variable in a list of LSTMStateTuples
        # to zero
        def get_state_reset_op(state_variables, cell, batch_size):
            zero_states = cell.zero_state(batch_size, tf.float32)
            return get_state_update_op(state_variables, zero_states)

        weights = {
            'z_mean': tf.get_variable(
                "z_mean",
                shape=[
                    self.intermediate_dim,
                    self.z_dim],
                initializer=tf.contrib.layers.xavier_initializer()),
            'log_sigma': tf.get_variable(
                "log_sigma",
                shape=[
                    self.intermediate_dim,
                    self.z_dim],
                initializer=tf.contrib.layers.xavier_initializer())}
        biases = {
            'z_mean_b': tf.get_variable("b_mean", shape=[self.z_dim],
                                        initializer=tf.zeros_initializer()),
            'z_std_b': tf.get_variable("b_log_sigma", shape=[self.z_dim],
                                       initializer=tf.zeros_initializer())
        }

        with tf.variable_scope("encoder"):
            with tf.variable_scope("LSTM_encoder"):
                lstm_layer = tf.nn.rnn_cell.LSTMCell(
                    self.intermediate_dim,
                    forget_bias=1,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.relu)

        if self.stateful:
            self.batch_ = tf.placeholder(tf.int32, shape=[])
            # throws an error without MultiRNNCell
            layer = tf.nn.rnn_cell.MultiRNNCell([lstm_layer])
            states = get_state_variables(self.batch_, layer)
            outputs, new_states = tf.nn.dynamic_rnn(
                layer, self.x, initial_state=states, dtype=tf.float32)
            self.update_op = get_state_update_op(states, new_states)
            self.reset_state_op = get_state_reset_op(
                states, lstm_layer, self.batch_)
        else:
            outputs, _ = tf.nn.dynamic_rnn(lstm_layer, self.x, dtype="float32")

        # For each layer, get the initial state. states will be a tuple of
        # LSTMStateTuples.
        self.z_mean = tf.add(tf.matmul(
            outputs[:, -1, :], weights['z_mean']), biases['z_mean_b'])
        self.z_sigma = tf.nn.softplus(tf.add(tf.matmul(
            outputs[:, -1, :], weights['log_sigma']), biases['z_std_b']))
        self.z = gauss_sampling(self.z_mean, self.z_sigma)

        # from [batch_size,z_dim] to [batch_size, TIMESTEPS, z_dim]
        repeated_z = tf.keras.layers.RepeatVector(
            self.repeat, dtype="float32")(self.z)

        with tf.variable_scope("decoder"):
            if self.stateful:
                with tf.variable_scope('lstm_decoder_stateful'):
                    rnn_layers_ = [
                        tf.nn.rnn_cell.LSTMCell(
                            size,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            forget_bias=1) for size in [
                            self.intermediate_dim,
                            n_dim]]
                    multi_rnn_cell_ = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_)
                    states_ = get_state_variables(self.batch_, multi_rnn_cell_)
                self.x_reconstr_mean, new_states_ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell_, inputs=repeated_z, initial_state=states_, dtype=tf.float32)
                self.update_op_ = get_state_update_op(states_, new_states_)
                self.reset_state_op_ = get_state_reset_op(
                    states_, multi_rnn_cell_, self.batch_)
            else:
                with tf.variable_scope('lstm_decoder_stateless'):
                    rnn_layers = [
                        tf.nn.rnn_cell.LSTMCell(
                            size,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            forget_bias=1) for size in [
                            self.intermediate_dim,
                            n_dim]]
                    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                self.x_reconstr_mean, _ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell, inputs=repeated_z, dtype=tf.float32)

    def _create_loss_optimizer(self, opt, **param):
        with tf.name_scope("MSE"):
            reconstr_loss = tf.reduce_sum(
                tf.losses.mean_squared_error(
                    self.x, self.x_reconstr_mean))
        with tf.name_scope("KL_divergence"):
            latent_loss = - 0.5 * tf.reduce_sum(1 + self.z_sigma
                                                - self.z_mean ** 2
                                                - tf.exp(self.z_sigma), 1)
            self._cost = tf.reduce_mean(reconstr_loss + self.kulback_coef * latent_loss)
        # apply gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), 10)
        self.train_op = opt(**param).apply_gradients(zip(grads, tvars))

    def fit(
            self,
            X,
            learning_rate=0.001,
            batch_size=100,
            num_epochs=200,
            opt=tf.train.AdamOptimizer,
            REG_LAMBDA=0,
            grad_clip_norm=10,
            optimizer_params=None,
            verbose=True):

        if len(np.shape(X)) != 3:
            raise ValueError(
                'Input must be a 3-D array. I could reshape it for you, but I am too lazy.'
                ' \n            Use input.reshape(-1,timesteps,1).')
        if optimizer_params is None:
            optimizer_params = {}
            optimizer_params['learning_rate'] = learning_rate
        else:
            optimizer_params = dict(six.iteritems(optimizer_params))

        self._create_loss_optimizer(opt, **optimizer_params)
        lstm_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='LSTM_encoder')
        self._cost += REG_LAMBDA * tf.reduce_mean(tf.nn.l2_loss(lstm_var))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X,
                self.batch_size: batch_size})
        batches_per_epoch = int(np.ceil(len(X) / batch_size))

        print("\n")
        print("Training...")
        print("\n")
        start = timer()

        for epoch in range(num_epochs):
            train_error = 0
            for step in range(batches_per_epoch):
                if self.stateful:
                    loss, _, s, _ = self.sess.run([self._cost, self.train_op, self.update_op, self.update_op_],
                                                  feed_dict={self.repeat: np.shape(X)[1], self.batch_: batch_size})
                else:
                    loss, _ = self.sess.run([self._cost, self.train_op], feed_dict={
                        self.repeat: np.shape(X)[1]})
                train_error += loss
            if step == (batches_per_epoch - 1):
                mean_loss = train_error / batches_per_epoch

                if self.stateful:  # reset cell & hidden states between epochs
                    self.sess.run([self.reset_state_op],
                                  feed_dict={self.batch_: batch_size})
                    self.sess.run([self.reset_state_op_],
                                  feed_dict={self.batch_: batch_size})
            if epoch % 10 == 0 & verbose:
                print(
                    "Epoch {:^6} Loss {:0.5f}".format(
                        epoch + 1, mean_loss))
        end = timer()
        print("\n")
        print("Training time {:0.2f} minutes".format((end - start) / (60)))

    def reconstruct(self, X, get_error=False):
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X,
                self.batch_size: np.shape(X)[0]})
        if self.stateful:
            _, _ = self.sess.run([self.reset_state_op, self.reset_state_op_], feed_dict={
                self.batch_: np.shape(X)[0]})
            x_rec, _, _ = self.sess.run([self.x_reconstr_mean, self.update_op, self.update_op_], feed_dict={
                self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]})
        else:
            x_rec = self.sess.run(self.x_reconstr_mean,
                                  feed_dict={self.repeat: np.shape(X)[1]})
        if get_error:
            squared_error = (x_rec - X) ** 2
            return x_rec, squared_error
        else:
            return x_rec

    def reduce(self, X):
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X,
                self.batch_size: np.shape(X)[0]})
        if self.stateful:
            _ = self.sess.run([self.reset_state_op], feed_dict={
                self.batch_: np.shape(X)[0]})
            x, _ = self.sess.run([self.z, self.update_op], feed_dict={
                self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]})
        else:
            x = self.sess.run(self.z)
        return x