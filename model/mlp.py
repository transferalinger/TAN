"""
File: model/mlp.py
 - Contain model class for mlp.
"""
import random, string
import tensorflow as tf
import numpy as np

from .model_base import ModelBase
from .common import input_pipeline_one_hot,\
                    make_directory


class MLP(ModelBase):
    """
    Description: model class for mlp.
    """

    def __init__(self,
                 network_architecture,
                 learning_rate,
                 optimizer_type,
                 batch_size,
                 data_file,
                 data_index,
                 label_index,
                 num_epochs,
                 n_instance,
                 log_dir,
                 use_pretrained,
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
        :param num_epochs: the number of epochs to train the model
        :param n_instance: the number of instances in the data file to train the model
        :param log_dir: the directory path to write logs during training
        :param use_pretrained: whether to initialize from pretrained mlp
        :param pretrain_path: the directory path of the pretrained mlp
        """

        # set configurations from arguments
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.n_instance = n_instance
        self.log_dir = log_dir
        self.use_pretrained = use_pretrained
        self.pretrain_path = pretrain_path

        # set log file information
        self.create_log_files('epoch,iter,cost,train_acc,test_acc,test_auc_roc,test_auc_pr')
        self.log_format = '%04d,%d,%.9f,%.9f,%.9f,%.9f,%.9f'

        # define input pipeline for reading features and labels
        self.x, self.y = input_pipeline_one_hot(
                data_file,
                delimiter=',',
                batch_size=batch_size,
                data_index=data_index,
                label_index=label_index,
                n_col=1+len(data_index),
                n_out = network_architecture["n_output"],
                num_epochs=num_epochs)

        #self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]], name="input")
        #self.y = tf.placeholder(tf.float32, [None, network_architecture["n_output"]])

        # create computation graph using tensorflow
        self.out = self._create_network()
        self._create_loss_optimizer()
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

    def _load_pretrained(self):
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

    def _initialize_weights(self):
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
        units = [self.network_architecture['n_input']] \
                + self.network_architecture['hidden']

        for i in range(n_layer):
            n = i + 1
            all_weights['weights']['w' + str(n)] = tf.Variable(
                    tf.random_normal([units[i], units[i+1]], stddev=2.0 / np.sqrt(units[i] + units[i+1])))
            all_weights['biases']['b' + str(n)] = tf.Variable(
                    tf.zeros([units[i+1]]))

        all_weights['weights']['out'] = tf.Variable(
                tf.random_normal([
                    units[-1],
                    self.network_architecture['n_output']], stddev=2.0 / np.sqrt(units[-1] + self.network_architecture['n_output'])))
        all_weights['biases']['out'] = tf.Variable(
                tf.zeros([self.network_architecture['n_output']]))

        return all_weights

    def _create_network(self):
        """
        Create mlp model.

        Returns
        -------
        tensorflow tensor object for mlp output
        """
        # Initialize autoencode network weights and biases
        if self.use_pretrained:
            self.network_weights = self._load_pretrained()
        else:
            self.network_weights = self._initialize_weights()

        weights = self.network_weights['weights']
        biases = self.network_weights['biases']

        layer = self.x
        n_layer = len(self.network_architecture['hidden'])

        #for i in range(len(self.network_weights)-1):
        for i in range(n_layer):
            n = i + 1
            layer = tf.nn.relu(tf.add(tf.matmul(layer, weights['w' + str(n)]), biases['b' + str(n)]))
        out = tf.add(tf.matmul(layer, weights['out']), biases['out'])
        return out

    def _create_loss_optimizer(self):
        """
        Define loss and optimizer for training.
        """
        #name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.y))

        if self.optimizer_type == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        elif self.optimizer_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)
        elif self.optimizer_type == 'adagradda':
            self.optimizer = tf.train.AdagradDAOptimizer(self.learning_rate).minimize(self.cost)
        elif self.optimizer_type == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)
        elif self.optimizer_type == 'ftrl':
            self.optimizer = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.cost)

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

        e_units = [self.network_architecture['n_input']] \
                + self.network_architecture['hidden']

        n_layer = len(e_units)
        # encoder
        for i in range(n_layer-1):
            n = i + 1
            h = self.network_weights['weights']['w' + str(n)]
            b = self.network_weights['biases']['b' + str(n)]
            dumps.append(self.sess.run([h, b]))

        out_w = self.network_weights['weights']['out']
        out_b = self.network_weights['biases']['out'] 
        dumps.append(self.sess.run([out_w, out_b]))
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
            np.savetxt(_dir + "/mlp_w" + str(n), dumps[k][0], delimiter=",")
            np.savetxt(_dir + "/mlp_b" + str(n), dumps[k][1], delimiter=",")
            k+=1

        np.savetxt(_dir + "/mlp_out_w", dumps[k][0], delimiter=",")
        np.savetxt(_dir + "/mlp_out_b", dumps[k][1], delimiter=",")
        return

    def learn(self, test_d, test_l, test_step, early_stop):
        """
        Starts training.

        :param test_d: test feature
        :param test_l: test label
        :param test_step: steps to calculate test predictions
        :param early_stop: maxumum step for early stopping
        """
        coord = tf.train.Coordinator()
        self.init_session()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            step = 0
            early_stop_step = 0
            best_auc = -np.inf
            best_record = []

            test_acc, test_auc_roc, test_auc_pr = self.predict(test_d, test_l)

            while not coord.should_stop():
                # Compute average loss
                opt, cost, train_acc = self.sess.run([self.optimizer, self.cost, self.accuracy])
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

        print("Best"),
        print(self.log_format % best_record)
        self.write_best(best_record)
        self.close_logs()

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
