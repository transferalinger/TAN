"""
File: model/auto_encoder.py
 - Contain model class for autoencoder.
"""
import tensorflow as tf
import numpy as np
import random

from .model_base import ModelBase
from .common import make_directory, \
                    input_pipeline


class Autoencoder(ModelBase):
    """
    Description: model class for autoencoder.
    """

    def __init__(self,
                 network_architecture,
                 learning_rate,
                 optimizer_type,
                 batch_size,
                 data_file,
                 data_index,
                 n_instance,
                 log_dir,
                 num_epochs,
                 reg_param,
                 use_pretrained,
                 pretrain_path=''):
        """
        Initialize models as argument configurations.

        :param network_architecture: architecture of autoencoder
        :param learning_rate: learning rate to train the model
        :param optimizer_type: optimizer to train the model
        :param batch_size: batch size to train the model
        :param data_file: the directory path of the data file to train the model
        :param data_index: the list of indices for features in the data file to train the model
        :param n_instance: the number of instances in the data file to train the model
        :param log_dir: the directory path to write logs during training
        :param num_epochs: the number of epochs to train the model
        :param reg_param: the regularization parameter for sparsity
        :param use_pretrained: whether to initialize from pretrained autoencoder
        :param pretrain_path: the directory path of the pretrained autoencoder
        """

        # set configurations from arguments
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.n_instance = n_instance
        self.log_dir = log_dir
        self.reg_param = reg_param
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

        # create computation graph using tensorflow
        self._create_network()
        self._create_loss_optimizer()
        self.saver = tf.train.Saver()
        return

    def init_session(self, device="cpu"):
        """
        Initialize session to use cpu or gpu and initialize variabels

        :param device: cpu or gpu for training
        """
        # Initializing the tensorflow variables.
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options)
        if device == "cpu":
            config = tf.ConfigProto(device_count={'GPU': 0})

        # Launch the session
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        return

    def _load_pretrained(self):
        """
        Load pretrained weights of pretrained autoencoder as dictionary.

        Returns
        -------
        dictionary of pretrained autoencoder weights
        """
        all_weights = dict()
        all_weights['weights_encoder'] = {}
        all_weights['biases_encoder'] = {}
        all_weights['weights_decoder'] = {}
        all_weights['biases_decoder'] = {}

        e_units = [self.network_architecture['n_input']]\
            + self.network_architecture['hidden']
        n_layer = len(e_units)
        weight_prefix = self.pretrain_path + '/weight'
        # encoder
        for i in range(n_layer-1):
            n = i + 1
            all_weights['weights_encoder']['w' + str(n)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/encoder_w' + str(n),
                               dtype=np.float32,
                               delimiter=','))
            all_weights['biases_encoder']['b' + str(n)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/encoder_b' + str(n),
                               dtype=np.float32,
                               delimiter=','))

        all_weights['weights_encoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/encoder_out_w',
                           dtype=np.float32,
                           delimiter=','))
        all_weights['biases_encoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/encoder_out_b',
                           dtype=np.float32,
                           delimiter=','))

        for i in range(n_layer-1):
            n = i + 1
            all_weights['weights_decoder']['w' + str(n)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/decoder_w' + str(n),
                               dtype=np.float32,
                               delimiter=','))
            all_weights['biases_decoder']['b' + str(n)] = tf.Variable(
                    np.loadtxt(weight_prefix + '/decoder_b' + str(n),
                               dtype=np.float32,
                               delimiter=','))

        all_weights['weights_decoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/decoder_out_w',
                           dtype=np.float32,
                           delimiter=','))
        all_weights['biases_decoder']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/decoder_out_b',
                           dtype=np.float32,
                           delimiter=','))

        return all_weights

    def _initialize_weights(self):
        """
        Initialize weights of autoencoder as dictionary.

        Returns
        -------
        dictionary of initialized autoencoder weights
        """
        all_weights = dict()
        all_weights['weights_encoder'] = {}
        all_weights['biases_encoder'] = {}
        all_weights['weights_decoder'] = {}
        all_weights['biases_decoder'] = {}

        e_units = [self.network_architecture['n_input']] + self.network_architecture['hidden']
        d_units = [self.network_architecture['n_z']] + self.network_architecture['hidden'][::-1]

        #print('e_units' , e_units)
        #print('d_units' , d_units)
        n_layer = len(e_units)
        # encoder
        for i in range(n_layer-1):
            n = i + 1
            all_weights['weights_encoder']['w' + str(n)] = tf.Variable(tf.random_normal([e_units[i], e_units[i+1]], stddev=2.0 / np.sqrt(e_units[i] + e_units[i+1])), name='en_w/h'+str(n))
            all_weights['biases_encoder']['b' + str(n)] = tf.Variable(tf.zeros([e_units[i+1]]), name='en_b/b'+str(n))

        all_weights['weights_encoder']['out'] = tf.Variable(tf.random_normal([e_units[-1], self.network_architecture['n_z']], stddev=2.0 / np.sqrt(e_units[-1] + self.network_architecture['n_z'])), name='en_w/out')
        all_weights['biases_encoder']['out'] = tf.Variable(tf.zeros([self.network_architecture['n_z']]), name='en_b/out')

        for i in range(n_layer-1):
            n = i + 1
            all_weights['weights_decoder']['w' + str(n)] = tf.Variable(tf.random_normal([d_units[i], d_units[i+1]], stddev=2.0 / np.sqrt(d_units[i] + d_units[i+1])), name='de_w/h'+str(n))
            all_weights['biases_decoder']['b' + str(n)] = tf.Variable(tf.zeros([d_units[i+1]]), name='de_b/b'+str(n))

        all_weights['weights_decoder']['out'] = tf.Variable(tf.random_normal([d_units[-1], self.network_architecture['n_input']], stddev=2.0 / np.sqrt(d_units[-1] + self.network_architecture['n_input'])), name='de_w/out')
        all_weights['biases_decoder']['out'] = tf.Variable(tf.zeros([self.network_architecture['n_input']]), name='de_b/out')

        return all_weights

    def _create_network(self):
        """
        Create autoencoder model.
        """
        # Initialize autoencode network weights and biases
        if self.use_pretrained:
            self.network_weights = self._load_pretrained()
        else:
            self.network_weights = self._initialize_weights()

        # Construct model
        self.encoder_op = self._encoder(self.network_weights['weights_encoder'], self.network_weights['biases_encoder'])
        self.decoder_op = self._decoder(self.network_weights['weights_decoder'], self.network_weights['biases_decoder'])
        return

    def _encoder(self, weights, biases):
        """
        Create encoder model from weights and bias parameter variables.

        :param weights: dictionary of weight variables
        :param biases: dictionary of bias variables

        Returns
        -------
        tensorflow tensor object for encoder output
        """
        # Building the encoder
        # Encoder Hidden layer with sigmoid activation #1
        layer = self.x
        for i in range(len(weights)-1):
            n = i + 1
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights['w' + str(n)]), biases['b' + str(n)]))
        z_out = tf.add(tf.matmul(layer, weights['out']), biases['out'])
        return z_out

    def _decoder(self, weights, biases):
        """
        Create decoder model from weights and bias parameter variables.

        :param weights: dictionary of weight variables
        :param biases: dictionary of bias variables

        Returns
        -------
        tensorflow tensor object for encoder output
        """
        # Building the decoder
        # Decoder Hidden layer with sigmoid activation #1
        layer = self.encoder_op
        for i in range(len(weights)-1):
            n = i + 1
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights['w' + str(n)]), biases['b' + str(n)]))
        out = tf.add(tf.matmul(layer, weights['out']), biases['out'])
        return out

    def _create_loss_optimizer(self):
        """
        Define loss and optimizer for training.
        """
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.pow(self.x - self.decoder_op, 2), 1))
        self.reg_cost = self.cost
        if self.reg_param > 0:
            self.reg_cost = self.reg_cost + self.reg_param * tf.reduce_mean(tf.norm(self.encoder_op, ord=1, axis=1))
        lr = self.learning_rate

        if self.optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(lr).minimize(self.reg_cost)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.reg_cost)
        elif self.optimizer_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(lr).minimize(self.reg_cost)
        elif self.optimizer_type == 'adagradda':
            self.optimizer = tf.train.AdagradDAOptimizer(lr).minimize(self.reg_cost)
        elif self.optimizer_type == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(lr).minimize(self.reg_cost)
        elif self.optimizer_type == 'ftrl':
            self.optimizer = tf.train.FtrlOptimizer(lr).minimize(self.reg_cost)

    def dump_weight(self):
        """
        Dump weights of the model as a list.

        Returns
        -------
        a list of weights of the model
        """
        dumps = []

        e_units = [self.network_architecture['n_input']] + self.network_architecture['hidden']
        d_units = [self.network_architecture['n_z']] + self.network_architecture['hidden'][::-1]

        n_layer = len(e_units)
        # encoder
        for i in range(n_layer-1):
            n = i + 1
            h = self.network_weights['weights_encoder']['w' + str(n)]
            b = self.network_weights['biases_encoder']['b' + str(n)]
            dumps.append(self.sess.run([h, b]))

        en_out_h = self.network_weights['weights_encoder']['out']
        en_out_b = self.network_weights['biases_encoder']['out']
        dumps.append(self.sess.run([en_out_h, en_out_b]))

        for i in range(n_layer-1):
            n = i + 1
            h = self.network_weights['weights_decoder']['w' + str(n)]
            b = self.network_weights['biases_decoder']['b' + str(n)]
            dumps.append(self.sess.run([h, b]))

        de_out_h = self.network_weights['weights_decoder']['out']
        de_out_b = self.network_weights['biases_decoder']['out']
        dumps.append(self.sess.run([de_out_h, de_out_b]))
        return dumps

    def save_weight(self, dumps, _dir):
        """
        Save a list of weights of the model into a file.

        :param dumps: a list of weights of the model
        :param _dir: directory path to save the weights
        """
        e_units = [self.network_architecture['n_input']] + self.network_architecture['hidden']
        n_layer = len(e_units)
        k = 0
        # encoder
        for i in range(n_layer-1):
            n = i+1
            np.savetxt(_dir + "/encoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/encoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        np.savetxt(_dir + "/encoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/encoder_out_b", dumps[k][1], delimiter=",")
        k+=1

        for i in range(n_layer-1):
            n = i+1
            np.savetxt(_dir + "/decoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/decoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        np.savetxt(_dir + "/decoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/decoder_out_b", dumps[k][1], delimiter=",")
        return

    def learn(self, test_d, display_step, early_stop, device):
        """
        Starts training.

        :param test_d: test feature
        :param display_step: steps to calculate test predictions
        :param early_stop: maxumum step for early stopping
        :param device: cpu or gpu for training
        """
        coord = tf.train.Coordinator()
        self.init_session(device)
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        best_record = []
        best_weight = []

        try:
            step = 0
            early_stop_step = 0
            best_loss = np.inf
            test_loss, test_error, test_normed_error = self.predict(test_d)

            while not coord.should_stop():
                opt, train_loss = self.sess.run([self.optimizer, self.cost])
                epoch = (step * self.batch_size) // self.n_instance

                if step % display_step == 0:
                    test_loss, test_error, test_normed_error = self.predict(test_d)
                    self.write_test((epoch+1, step, train_loss, test_loss, test_error, test_normed_error))

                    if test_loss < best_loss:
                        early_stop_step = 0
                        best_loss = test_loss
                        best_record = (epoch+1, step, train_loss, test_loss, test_error, test_normed_error)
                        best_weight = self.dump_weight()
                    #else:
                        #early_stop_step += 1

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
        [loss, error, orig] = self.sess.run([self.cost,
                                             tf.sqrt(tf.reduce_sum(tf.pow(self.decoder_op - self.x, 2), axis=1)),
                                             tf.sqrt(tf.reduce_sum(tf.pow(self.x, 2), axis=1))],
                                            feed_dict={self.x: X})
        normed_error = error / orig
        return loss, np.mean(error), np.mean(normed_error)

    def predict_decoder(self, X):
        """
        Output autoencoder reconstruction from feature

        :param X: feature

        Returns
        -------
        autoencoder reconsturction from the feature
        """
        decoder_output = self.sess.run(self.decoder_op, feed_dict={self.x: X})
        return decoder_output

    def save(self, name):
        """
        Save the tensorflow objects into a file.

        :param name: directory path to save file
        """
        saver_path = self.saver.save(self.sess, name)
        print(saver_path)
