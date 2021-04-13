"""
File: model/match_features.py
 - Contain model class for transfer normalization.
"""
import random
import json
import tensorflow as tf
import numpy as np
import os

from .model_base import ModelBase
from .common import make_directory, \
                    input_pipeline


def load_ae_architecture(path, is_ae=False):
    """
    Read hyperparameter configurations from pretrained models.

    :param path: path to read configurations
    :param is_ae: if true read ae hyperparameter file else common hyperparameter file

    Returns
    -------
    hyperparameter configurations for previously trained models
    """
    if is_ae and os.path.exists(path + '/ae_hyperparameter'):
        with open(path + '/ae_hyperparameter') as f:
            hyper = json.loads(f.readline())
        return hyper
    else:
        with open(path + '/hyperparameter') as f:
            hyper = json.loads(f.readline())
        return hyper

class Matchnet(ModelBase):
    """
    Description: model class for trasnfer noramlization.
    """

    def __init__(self,
                 network_architecture,
                 learning_rate,
                 optimizer_type,
                 batch_size,
                 data_file,
                 data_index,
                 label_index,
                 n_instance,
                 log_dir,
                 num_epochs,
                 encoder1_path,
                 encoder2_path,
                 encoder_lr,
                 reg_layer_idx,
                 reg_param,
                 fix_pretrained,
                 use_pretrained,
                 pretrain_path):
        """
        Initialize models as argument configurations.

        :param network_architecture: architecture of transfer normalization layer
        :param learning_rate: learning rate to train the model
        :param optimizer_type: optimizer to train the model
        :param batch_size: batch size to train the model
        :param data_file: the directory path of the data file to train the model
        :param data_index: the list of indices for features in the data file to train the model
        :param label_index: the index for label in the data file to train the model
        :param n_instance: the number of instances in the data file to train the model
        :param log_dir: the directory path to write logs during training
        :param num_epochs: the number of epochs to train the model
        :param encoder1_path: the path for the pretrained source autoencoder
        :param encoder2_path: the path for the pretrained target autoencoder
        :param encoder_lr: the learning rate for encoder training
        :param reg_layer_idx: the transfer normalization layer index to apply regularization
        :param reg_param: the regularization parameter for sparsity
        :param fix_pretrained: whether to fix parameters of pretrained transfer normalization layer
        :param use_pretrained: whether to initialize from pretrained transfer normalization layer
        :param pretrain_path: the directory path of the pretrained transfer normalization layer
        """

        # set configurations from arguments
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.n_instance = n_instance
        self.log_dir = log_dir
        self.encoder1_path = encoder1_path
        self.encoder2_path = encoder2_path
        self.encoder_lr = encoder_lr
        self.reg_layer_idx = reg_layer_idx
        self.reg_param = reg_param
        self.pretrain_trainable = not fix_pretrained
        self.use_pretrained = use_pretrained
        self.pretrain_path = pretrain_path

        # set log file information
        self.create_log_files('epoch,iter,train_loss,test_loss,test_error,test_normed_error')
        self.log_format = '%04d,%d,%.9f,%.9f,%.9f,%.9f'

        # define input pipeline for reading features
        self.x = input_pipeline(data_file,
                                delimiter=',',
                                batch_size=batch_size,
                                data_index=data_index,
                                n_col=1+len(data_index),
                                num_epochs=num_epochs)

        # create computation graph of encoders using tensorflow
        self.ae1_hyper = load_ae_architecture(self.encoder1_path, is_ae=True)
        self.ae2_hyper = load_ae_architecture(self.encoder2_path, is_ae=False)
        with tf.variable_scope("ae1"):
            self.ae1_weights = self.load_ae_weights(self.encoder1_path, self.ae1_hyper)
        with tf.variable_scope("ae2"):
            self.ae2_weights = self.load_ae_weights(self.encoder2_path, self.ae2_hyper)
        self.encoder1 = self.create_encoder(self.ae1_weights)
        self.encoder2 = self.create_encoder(self.ae2_weights)

        # create computation graph of transfer normalization layer using tensorflow
        with tf.variable_scope("mn"):
            self.out = self.create_network()
        self.create_loss_optimizer()
        self.saver = tf.train.Saver()
        return

    def init_session(self):
        """
        Initialize session and tensorflow variabels
        """
        # Initializing the tensorflow variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        return

    def load_ae_weights(self, encoder_path, ae_hyper):
        """
        Load pretrained weights of pretrained autoencoder as dictionary.

        :param encoder_path: directory path of pretrained autoencoder
        :param ae_hyper: dictionary of hyperparameter for autoencoder

        Returns
        -------
        dictionary of pretrained autoencoder weights
        """
        all_weights = dict()
        all_weights['weights_encoder'] = {}
        all_weights['biases_encoder'] = {}
        all_weights['weights_decoder'] = {}
        all_weights['biases_decoder'] = {}
        weight_prefix = encoder_path + '/weight'

        # Load encoder hidden layers
        for i in range(1, len(ae_hyper['hidden']) + 1):
            all_weights['weights_encoder']['w' + str(i)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/encoder_w' +str(i),
                               dtype=np.float32,
                               delimiter=','),
                    trainable=self.pretrain_trainable)
            all_weights['biases_encoder']['b' + str(i)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/encoder_b' +str(i),
                               dtype=np.float32,
                               delimiter=','),
                    trainable=self.pretrain_trainable)

        # Load encoder out layer
        all_weights['weights_encoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/encoder_out_w',
                           dtype=np.float32,
                           delimiter=','),
                trainable=self.pretrain_trainable)
        all_weights['biases_encoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/encoder_out_b',
                           dtype=np.float32,
                           delimiter=','),
                trainable=self.pretrain_trainable)

        for i in range(1, len(ae_hyper['hidden']) + 1):
            all_weights['weights_decoder']['w' + str(i)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/decoder_w' + str(i),
                               dtype=np.float32,
                               delimiter=','),
                    trainable=self.pretrain_trainable)
            all_weights['biases_decoder']['b' + str(i)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/decoder_b' + str(i),
                               dtype=np.float32,
                               delimiter=','),
                    trainable=self.pretrain_trainable)

        all_weights['weights_decoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/decoder_out_w',
                           dtype=np.float32,
                           delimiter=','),
                trainable=self.pretrain_trainable)
        all_weights['biases_decoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/decoder_out_b',
                           dtype=np.float32,
                           delimiter=','),
                trainable=self.pretrain_trainable)

        return all_weights

    def load_pretrained(self):
        """
        Load pretrained weights of pretrained transfer normalization layer as dictionary.

        Returns
        -------
        dictionary of pretrained transfer normalization layer weights
        """
        all_weights = dict()
        all_weights['weights'] = {}
        all_weights['biases'] = {}

        n_layer = len(self.network_architecture['hidden'])
        units = [self.ae2_hyper['z']] + self.network_architecture['hidden']
        print(units)
        weight_prefix = self.pretrain_path + '/weight'

        for i in range(len(units)-1):
            print(units[i])
            all_weights['weights']['w' + str(i+1)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/mn_w' + str(i+1),
                               dtype=np.float32,
                               delimiter=','))
            all_weights['biases']['b' + str(i+1)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/mn_b' + str(i+1),
                               dtype=np.float32,
                               delimiter=','))

        all_weights['weights']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/mn_out_w',
                           dtype=np.float32,
                           delimiter=','))
        all_weights['biases']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/mn_out_b',
                           dtype=np.float32,
                           delimiter=','))

        return all_weights

    def initialize_weights(self):
        """
        Initialize weights of transfer normalization layer as dictionary.

        Returns
        -------
        dictionary of initialized transfer normalization layer weights
        """
        all_weights = dict()
        all_weights['weights'] = {}
        all_weights['biases'] = {}

        n_layer = len(self.network_architecture['hidden'])
        units = [self.ae2_hyper['z']] + self.network_architecture['hidden']
        print(units)

        for i in range(len(units)-1):
            print(units[i])
            all_weights['weights']['w' + str(i+1)] = tf.Variable(tf.random_normal([units[i], units[i+1]], stddev=2.0 / np.sqrt(units[i] + units[i+1])))
            all_weights['biases']['b' + str(i+1)] = tf.Variable(tf.zeros([units[i+1]]))

        all_weights['weights']['out'] = tf.Variable(tf.random_normal([units[-1], self.ae1_hyper['z']], stddev=2.0 / np.sqrt(units[-1] + self.ae1_hyper['z'])))
        all_weights['biases']['out'] = tf.Variable(tf.zeros([self.ae1_hyper['z']]))

        return all_weights

    def create_encoder(self, ae_weights):
        """
        Create autoencoder model.

        :param ae_weights: dictionary of pretrained autoencoder weights

        Returns
        -------
        tensorflow tensor object for encoder output
        """
        weights = ae_weights['weights_encoder']
        biases = ae_weights['biases_encoder']
        # Build the encoder
        layer = self.x
        for i in range(len(weights)-1):
            n = i + 1
            layer = tf.nn.relu(
                    tf.add(
                        tf.matmul(layer, weights['w' + str(n)]),
                        biases['b' + str(n)]))
        z_out = tf.add(
                    tf.matmul(layer, weights['out']),
                    biases['out'])
        return z_out

    def create_network(self):
        """
        Create transfer normalization model.

        Returns
        -------
        tensorflow tensor object for transfer normalization output
        """
        if self.use_pretrained:
            self.network_weights = self.load_pretrained()
        else:
            self.network_weights = self.initialize_weights()

        # Initialize autoencode network weights and biases
        weights = self.network_weights['weights']
        biases = self.network_weights['biases']

        layer = self.encoder2
        for n in range(1, len(self.network_architecture['hidden'])+1):
            print(layer)
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights['w' + str(n)]), biases['b' + str(n)]))
            if n == self.reg_layer_idx:
                self.reg_layer = layer
        z_out = tf.add(tf.matmul(layer, weights['out']), biases['out'])

        return z_out

    def create_loss_optimizer(self):
        """
        Define loss and optimizer for training.
        """
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.pow(self.encoder1 - self.out, 2), 1))
        self.reg_cost = self.cost
        if self.reg_param > 0:
            self.reg_cost = self.reg_cost + self.reg_param * tf.reduce_mean(tf.norm(self.reg_layer, ord=1, axis=1))

        ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ae")
        mn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mn")
        print(ae_vars)
        print(mn_vars)

        if self.optimizer_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer
        elif self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer
        elif self.optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer
        elif self.optimizer_type == 'adagradda':
            optimizer = tf.train.AdagradDAOptimizer
        elif self.optimizer_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer
        elif self.optimizer_type == 'ftrl':
            optimizer = tf.train.FtrlOptimizer

        if self.pretrain_trainable:
            self.ae_optimizer = optimizer(self.encoder_lr).minimize(
                    self.reg_cost,
                    var_list=ae_vars)
        else:
            self.ae_optimizer = None

        self.mn_optimizer = optimizer(self.learning_rate).minimize(
                self.reg_cost,
                var_list=mn_vars)

    def dump_weight(self):
        """
        Dump weights of the model as a list.

        Returns
        -------
        a list of weights of the model
        """
        dumps = []

        e1_units = [self.ae1_hyper['input']] + self.ae1_hyper['hidden']
        e2_units = [self.ae2_hyper['input']] + self.ae2_hyper['hidden']
        m_units = [self.ae2_hyper['z']] + self.network_architecture['hidden']

        # Encoder1 hidden layers
        n_layer = len(e1_units)
        for n in range(1, n_layer):
            w = self.ae1_weights['weights_encoder']['w' + str(n)]
            b = self.ae1_weights['biases_encoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        en_out_w = self.ae1_weights['weights_encoder']['out']
        en_out_b = self.ae1_weights['biases_encoder']['out']
        dumps.append(self.sess.run([en_out_w, en_out_b]))

        n_layer = len(e1_units)
        for n in range(1, n_layer):
            w = self.ae1_weights['weights_decoder']['w' + str(n)]
            b = self.ae1_weights['biases_decoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        de_out_w = self.ae1_weights['weights_decoder']['out']
        de_out_b = self.ae1_weights['biases_decoder']['out']
        dumps.append(self.sess.run([de_out_w, de_out_b]))

        # Encoder2 hidden layers
        n_layer = len(e2_units)
        for n in range(1, n_layer):
            w = self.ae2_weights['weights_encoder']['w' + str(n)]
            b = self.ae2_weights['biases_encoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        en_out_w = self.ae2_weights['weights_encoder']['out']
        en_out_b = self.ae2_weights['biases_encoder']['out']
        dumps.append(self.sess.run([en_out_w, en_out_b]))

        n_layer = len(e2_units)
        for n in range(1, n_layer):
            w = self.ae2_weights['weights_decoder']['w' + str(n)]
            b = self.ae2_weights['biases_decoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        de_out_w = self.ae2_weights['weights_decoder']['out']
        de_out_b = self.ae2_weights['biases_decoder']['out']
        dumps.append(self.sess.run([de_out_w, de_out_b]))

        # MN hidden layers
        n_layer = len(m_units)
        for n in range(1, n_layer):
            w = self.network_weights['weights']['w' + str(n)]
            b = self.network_weights['biases']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        mn_out_w = self.network_weights['weights']['out']
        mn_out_b = self.network_weights['biases']['out']
        dumps.append(self.sess.run([mn_out_w, mn_out_b]))
        return dumps

    def save_weight(self, dumps, _dir):
        """
        Save a list of weights of the model into a file.

        :param dumps: a list of weights of the model
        :param _dir: directory path to save the weights
        """
        k = 0

        # Save encoder1 hidden layers
        n_layer = len(self.ae1_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/encoder1_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/encoder1_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save encoder1 output layer
        np.savetxt(_dir + "/encoder1_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/encoder1_out_b", dumps[k][1], delimiter=",")
        k+=1

        # Save decoder1 hidden layers
        n_layer = len(self.ae1_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/decoder1_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/decoder1_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save decoder1 output layer
        np.savetxt(_dir + "/decoder1_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/decoder1_out_b", dumps[k][1], delimiter=",")
        k+=1

        # Save encoder2 hidden layers
        n_layer = len(self.ae2_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/encoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/encoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save encoder2 output layer
        np.savetxt(_dir + "/encoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/encoder_out_b", dumps[k][1], delimiter=",")
        k+=1

        # Save decoder2 hidden layers
        n_layer = len(self.ae2_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/decoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/decoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save decoder2 output layer
        np.savetxt(_dir + "/decoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/decoder_out_b", dumps[k][1], delimiter=",")
        k+=1

        # Save MN hidden layers
        n_layer = len(self.network_architecture['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/mn_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/mn_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save MN out layer
        np.savetxt(_dir + "/mn_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/mn_out_b", dumps[k][1], delimiter=",")

        return

    def learn(self, test_d, test_step, early_stop, device):
        """
        Starts training.

        :param test_d: test feature
        :param test_step: steps to calculate test predictions
        :param early_stop: maxumum step for early stopping
        :param device: cpu or gpu for training
        """
        coord = tf.train.Coordinator()
        self.init_session()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        best_record = []
        best_weight = []

        try:
            step = 0
            early_stop_step = 0
            best_loss = np.inf
            test_loss, test_error, test_normed_error = self.predict(test_d)

            while not coord.should_stop():
                # Compute average loss
                if self.pretrain_trainable:
                    (opt_ae, opt_mn, train_loss) = self.sess.run(
                            [self.ae_optimizer,
                            self.mn_optimizer,
                            self.cost])
                else:
                    (opt_mn, train_loss) = self.sess.run(
                            [self.mn_optimizer,
                            self.cost])
                epoch = (step * self.batch_size) // self.n_instance

                if step % test_step == 0:
                    test_loss, test_error, test_normed_error = self.predict(test_d)
                    self.write_test((epoch+1, step, train_loss, test_loss, test_error, test_normed_error))

                    if test_loss < best_loss:
                        early_stop_step = 0
                        best_loss = test_loss
                        best_record = (epoch+1, step, train_loss, test_loss, test_error, test_normed_error)
                        best_weight = self.dump_weight()
                    #else:
                        #early_stop_step+=1

                    log = self.log_format % (epoch+1, step, train_loss, test_loss, test_error, test_normed_error)
                    print(log)

                #if (early_stop_step >= early_stop):
                    #print('Done training -- early stop')
                    #coord.request_stop()

                self.write_train((epoch+1, step, train_loss, test_loss, test_error, test_normed_error))
                #self.flush_logs()
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done epoch')
        finally:
            coord.request_stop()
        coord.join(threads)

        print("Best")
        print(self.log_format % best_record)
        self.write_best(best_record)
        self.close_logs()
        
        hyper = open(self.log_dir + '/ae_hyperparameter', 'w')
        ae_hyper = dict(input=self.ae2_hyper['input'],
                        hidden=self.ae2_hyper['hidden'],
                        z=self.ae2_hyper['z'],
                        output=self.ae1_hyper['z'])
        hyper.write(json.dumps(ae_hyper))
        hyper.close()
        
        weight_dir = self.log_dir+"/weight"
        model_dir = self.log_dir+"/model"
        make_directory([weight_dir, model_dir])
        self.save_weight(best_weight, weight_dir)
        #self.save(model_dir + "/model.ckpt")

    def predict(self, X):
        """
        Output accuracy of model predictions on given features and labels

        :param X: feature

        Returns
        -------
        accuracy of model predictions
        """
        loss, error, orig = self.sess.run([self.cost,
                                           tf.sqrt(tf.reduce_sum(tf.pow(self.encoder1 - self.out, 2), axis=1)),
                                           tf.sqrt(tf.reduce_sum(tf.pow(self.encoder1, 2), axis=1))],
                                          feed_dict={self.x: X})
        normed_error = error / orig
        return loss, np.mean(error), np.mean(normed_error)

    def predict_encoder1(self, X):
        """
        Output source autoencoder reconstruction from feature

        :param X: feature

        Returns
        -------
        source autoencoder reconsturction from the feature
        """
        encoder_output = self.sess.run(self.encoder1, feed_dict={self.x: X})
        return encoder_output

    def predict_encoder2(self, X):
        """
        Output target autoencoder reconstruction from feature

        :param X: feature

        Returns
        -------
        target autoencoder reconsturction from the feature
        """
        encoder_output = self.sess.run(self.encoder2, feed_dict={self.x: X})
        return encoder_output

    def predict_matchnet(self, X):
        """
        Output transfer normalization reconstruction from feature

        :param X: feature

        Returns
        -------
        transfer normalization reconsturction from the feature
        """
        matchnet_output = self.sess.run(self.out, feed_dict={self.x: X})
        return matchnet_output

    def save(self, name):
        """
        Save the tensorflow objects into a file.

        :param name: directory path to save file
        """
        save_path = self.saver.save(self.sess, name)
        print(save_path)
