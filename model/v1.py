"""
File: model/v1.py
 - Contain model class for mlp on top of autoencoder.
"""
import random
import json
import tensorflow as tf
import numpy as np

from .model_base import ModelBase
from .common import make_directory, \
                    input_pipeline_one_hot


def load_ae_architecture(path):
    """
    Read hyperparameter configurations from pretrained models.

    :param path: path to read configurations

    Returns
    -------
    hyperparameter configurations for previously trained models
    """
    with open(path + '/hyperparameter') as f:
        hyper = json.loads(f.readline())
    return hyper


class V1(ModelBase):
    """
    Description: model class for mlp on top of autoencoder.
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
                 encoder_path,
                 encoder_lr,
                 use_pretrained,
                 fix_pretrained,
                 pretrain_path=''):
        """
        Initialize models as argument configurations.

        :param network_architecture: architecture of mlp
        :param learning_rate: learning rate to train the model
        :param optimizer_type: optimizer to train the model
        :param batch_size: batch size to train the model
        :param data_file: the directory path of the data file to train the model
        :param data_index: the list of indices for features in the data file to train the model
        :param label_index: the index for label in the data file to train the model
        :param n_instance: the number of instances in the data file to train the model
        :param log_dir: the directory path to write logs during training
        :param num_epochs: the number of epochs to train the model
        :param encoder_path: the path for the pretrained target autoencoder
        :param encoder_lr: the learning rate for encoder training
        :param use_pretrained: whether to initialize from pretrained mlp
        :param fix_pretrained: whether to fix parameters of pretrained mlp
        :param pretrain_path: the directory path of the pretrained mlp
        """

        # set configurations from arguments
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.n_instance = n_instance
        self.log_dir = log_dir
        self.encoder_path = encoder_path
        self.encoder_lr = encoder_lr
        self.use_pretrained = use_pretrained
        self.pretrain_trainable = not fix_pretrained
        self.pretrain_path = pretrain_path

        # set log file information
        self.create_log_files('epoch,iter,cost,train_acc,test_acc,test_auc_roc,test_auc_pr')
        self.log_format = '%04d,%d,%.9f,%.9f,%.9f,%.9f,%.9f'

        # define input pipeline for reading features and labels
        self.x, self.y = input_pipeline_one_hot(data_file,
                                        delimiter=',',
                                        batch_size=batch_size,
                                        data_index=data_index,
                                        label_index=label_index,
                                        n_col=1+len(data_index),
                                        n_out = network_architecture["n_output"],
                                        num_epochs=num_epochs)

        # create computation graph of autoencoder using tensorflow
        self.ae_hyper = load_ae_architecture(self.encoder_path)
        with tf.variable_scope("ae"):
            self.ae_weights = self.load_ae_weights()
        self.encoder = self.create_encoder()

        # create computation graph of mlp using tensorflow
        with tf.variable_scope("mlp"):
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

    def load_ae_weights(self):
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
        weight_prefix = self.encoder_path + '/weight'

        # Load encoder hidden layers
        for i in range(1, len(self.ae_hyper['hidden']) + 1):
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

        for i in range(1, len(self.ae_hyper['hidden']) + 1):
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

    def load_mlp_weights(self):
        """
        Load pretrained weights of pretrained mlp as dictionary.

        Returns
        -------
        dictionary of pretrained mlp weights
        """
        all_weights = dict()
        all_weights['weights'] = {}
        all_weights['biases'] = {}

        e_units = [self.network_architecture['n_input']]\
            + self.network_architecture['hidden']
        n_layer = len(e_units)
        weight_prefix = self.pretrain_path + '/weight'
        # encoder
        with tf.variable_scope("mlp"):
            for i in range(n_layer-1):
                n = i + 1
                all_weights['weights']['w' + str(n)] = tf.Variable(
                        np.loadtxt(weight_prefix + '/mlp_w' + str(n),
                                   dtype=np.float32,
                                   delimiter=','))
                all_weights['biases']['b' + str(n)] = tf.Variable(
                        np.loadtxt(weight_prefix + '/mlp_b' + str(n),
                                   dtype=np.float32,
                                   delimiter=','))

        all_weights['weights']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/mlp_out_w',
                           dtype=np.float32,
                           delimiter=','))
        all_weights['biases']['out'] = tf.Variable(
                np.loadtxt(weight_prefix + '/mlp_out_b',
                           dtype=np.float32,
                           delimiter=','))

        return all_weights

    def initialize_weights(self):
        """
        Initialize weights of mlp as dictionary.

        Returns
        -------
        dictionary of initialized mlp weights
        """
        all_weights = dict()
        all_weights['weights'] = {}
        all_weights['biases'] = {}

        n_layer = len(self.network_architecture['hidden'])
        units = [self.ae_hyper['z']] + self.network_architecture['hidden']
        print(units)

        for i in range(len(units)-1):
            print(units[i])
            all_weights['weights']['w' + str(i+1)] = tf.Variable(tf.random_normal([units[i], units[i+1]], stddev=2.0 / np.sqrt(units[i] + units[i+1])))
            all_weights['biases']['b' + str(i+1)] = tf.Variable(tf.zeros([units[i+1]]))

        all_weights['weights']['out'] = tf.Variable(tf.random_normal([units[-1], self.network_architecture['n_output']], stddev=2.0 / np.sqrt(units[-1] + self.network_architecture['n_output'])))
        all_weights['biases']['out'] = tf.Variable(tf.zeros([self.network_architecture['n_output']]))

        return all_weights

    def create_encoder(self):
        """
        Create autoencoder model.

        Returns
        -------
        tensorflow tensor object for encoder output
        """
        weights = self.ae_weights['weights_encoder']
        biases = self.ae_weights['biases_encoder']
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
        Create mlp model.

        Returns
        -------
        tensorflow tensor object for mlp output
        """
        if self.use_pretrained:
            self.network_weights = self.load_mlp_weights()
        else:
            self.network_weights = self.initialize_weights()

        # Initialize autoencode network weights and biases
        weights = self.network_weights['weights']
        biases = self.network_weights['biases']

        layer = self.encoder
        for n in range(1, len(self.network_architecture['hidden'])+1):
            print(layer)
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights['w' + str(n)]), biases['b' + str(n)]))
        z_out = tf.add(tf.matmul(layer, weights['out']), biases['out'])

        return z_out

    def create_loss_optimizer(self):
        """
        Define loss and optimizer for training.
        """
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y))

        ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ae")
        mlp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "mlp")
        print(ae_vars)
        print(mlp_vars)

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
                    self.cost,
                    var_list=ae_vars)
        else:
            self.ae_optimizer = None

        self.mlp_optimizer = optimizer(self.learning_rate).minimize(
                self.cost,
                var_list=mlp_vars)

        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.out,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        if self.network_architecture["n_output"] == 2:
            self.auc_roc, self.auc_roc_update = tf.metrics.auc(tf.argmax(self.y, 1), tf.argmax(self.out, 1), curve='ROC')
            self.auc_pr, self.auc_pr_update = tf.metrics.auc(tf.argmax(self.y, 1), tf.argmax(self.out, 1), curve='PR')
        return

    def dump_weight(self):
        """
        Dump weights of the model as a list.

        Returns
        -------
        a list of weights of the model
        """
        dumps = []

        e_units = [self.ae_hyper['input']] + self.ae_hyper['hidden']
        m_units = [self.ae_hyper['z']] + self.network_architecture['hidden']

        # Encoder hidden layers
        n_layer = len(e_units)
        for n in range(1, n_layer):
            w = self.ae_weights['weights_encoder']['w' + str(n)]
            b = self.ae_weights['biases_encoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        en_out_w = self.ae_weights['weights_encoder']['out']
        en_out_b = self.ae_weights['biases_encoder']['out']
        dumps.append(self.sess.run([en_out_w, en_out_b]))

        n_layer = len(e_units)
        for n in range(1, n_layer):
            w = self.ae_weights['weights_decoder']['w' + str(n)]
            b = self.ae_weights['biases_decoder']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        de_out_w = self.ae_weights['weights_decoder']['out']
        de_out_b = self.ae_weights['biases_decoder']['out']
        dumps.append(self.sess.run([de_out_w, de_out_b]))

        # MLP hidden layers
        n_layer = len(m_units)
        for n in range(1, n_layer):
            w = self.network_weights['weights']['w' + str(n)]
            b = self.network_weights['biases']['b' + str(n)]
            dumps.append(self.sess.run([w, b]))

        mlp_out_w = self.network_weights['weights']['out']
        mlp_out_b = self.network_weights['biases']['out']
        dumps.append(self.sess.run([mlp_out_w, mlp_out_b]))
        return dumps

    def save_weight(self, dumps, _dir):
        """
        Save a list of weights of the model into a file.

        :param dumps: a list of weights of the model
        :param _dir: directory path to save the weights
        """
        k = 0

        # Save encoder hidden layers
        n_layer = len(self.ae_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/encoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/encoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save encoder output layer
        np.savetxt(_dir + "/encoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/encoder_out_b", dumps[k][1], delimiter=",")
        k+=1

        n_layer = len(self.ae_hyper['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/decoder_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/decoder_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save decoder output layer
        np.savetxt(_dir + "/decoder_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/decoder_out_b", dumps[k][1], delimiter=",")
        k+=1

        # Save MLP hidden layers
        n_layer = len(self.network_architecture['hidden'])
        for i in range(n_layer):
            n = i+1
            np.savetxt(_dir + "/mlp_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/mlp_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        # Save MLP out layer
        np.savetxt(_dir + "/mlp_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/mlp_out_b", dumps[k][1], delimiter=",")

        return

    def learn(self, test_d, test_l, test_step, early_stop, device):
        """
        Starts training.

        :param test_d: test feature
        :param test_l: test label
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
            best_auc = -np.inf
            test_acc, test_auc_roc, test_auc_pr = self.predict(test_d, test_l)

            while not coord.should_stop():
                # Compute average loss
                if self.pretrain_trainable:
                    (opt_ae, opt_mlp, cost, train_acc) = self.sess.run(
                            [self.ae_optimizer,
                            self.mlp_optimizer,
                            self.cost,
                            self.accuracy])
                else:
                    (opt_mlp, cost, train_acc) = self.sess.run(
                            [self.mlp_optimizer,
                            self.cost,
                            self.accuracy])
                epoch = (step * self.batch_size) // self.n_instance

                if step % test_step == 0:
                    test_acc, test_auc_roc, test_auc_pr = self.predict(test_d, test_l)
                    self.write_test((epoch+1, step, cost, train_acc, test_acc, test_auc_roc, test_auc_pr))

                    if test_auc_pr > best_auc:
                        early_stop_step = 0
                        best_auc = test_auc_pr
                        best_record = (epoch+1, step, cost, train_acc, test_acc, test_auc_roc, test_auc_pr)
                        best_weight = self.dump_weight()
                    #else:
                        #early_stop_step+=1

                    log = self.log_format % (epoch+1, step, cost, train_acc, test_acc, test_auc_roc, test_auc_pr)
                    print(log)

                    #if early_stop_step >= early_stop:
                        #print('Done training -- early stop')
                        #coord.request_stop()

                self.write_train((epoch+1, step, cost, train_acc, test_acc, test_auc_roc, test_auc_pr))
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
        ae_hyper = dict(input=self.ae_hyper['input'],
                        hidden=self.ae_hyper['hidden'],
                        z=self.ae_hyper['z'])
        hyper.write(json.dumps(ae_hyper))
        hyper.close()

        weight_dir = self.log_dir+"/weight"
        model_dir = self.log_dir+"/model"
        make_directory([weight_dir, model_dir])
        self.save_weight(best_weight, weight_dir)
        #self.save(model_dir + "/model.ckpt")

    def predict(self, X, Y):
        """
        Output accuracy of model predictions on given features and labels

        :param X: feature
        :param Y: label

        Returns
        -------
        accuracy of model predictions
        """
        y = self.sess.run(tf.one_hot(tf.squeeze(tf.cast(Y, tf.int32)), self.network_architecture['n_output']))
        if self.network_architecture["n_output"] == 2:
            (acc, auc_roc, _, auc_pr, _) = self.sess.run(
                [self.accuracy, self.auc_roc, self.auc_roc_update, self.auc_pr, self.auc_pr_update],
                feed_dict={self.x: X, self.y: y})
        else:
            acc = self.sess.run(self.accuracy, feed_dict={self.x: X, self.y: y})
            auc_roc = acc
            auc_pr = acc
        return acc, auc_roc, auc_pr

    def save(self, name):
        """
        Save the tensorflow objects into a file.

        :param name: directory path to save file
        """
        save_path = self.saver.save(self.sess, name)
        print(save_path)
