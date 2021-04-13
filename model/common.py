"""
File: model/common.py
 - Contain common functions used in model classes.
"""
import tensorflow as tf
import numpy as np
import os


def make_directory(dir_list):
    """
    Makes directories on given list.

    :param dir_list: list of direcotry to make
    """
    for _dir in dir_list:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    return


def load_data(fname_list, delimiter, in_index, out_index):
    """
    Load feature and label from given list of files.

    :param fname_list: list of files to read data
    :param delimiter: delimiter to parse data
    :param in_index: list of feature indices
    :param out_index: label index

    Returns
    -------
    feature and label arrays
    """
    _in = [[] for i in range(len(in_index))]
    _out = []
    for fname in fname_list:
        with open(fname) as f:
            for line in f:
                line = line.strip().split(delimiter)
                for i in range(len(in_index)):
                    _in[i].append(np.float(line[in_index[i]]))
                _out.append(np.float(line[out_index]))
    return np.array(_in).T, np.array([_out])


def load_data_no_out(fname_list, delimiter, in_index):
    """
    Load feature from given list of files.

    :param fname_list: list of files to read data
    :param delimiter: delimiter to parse data
    :param in_index: list of feature indices

    Returns
    -------
    feature array
    """
    _in = [[] for i in range(len(in_index))]
    for fname in fname_list:
        with open(fname) as f:
            for line in f:
                line = line.strip().split(delimiter)
                for i in range(len(in_index)):
                    _in[i].append(np.float(line[in_index[i]]))
    return np.array(_in).T


def create_file_reader_ops(filename_queue,
                           delimiter,
                           data_index,
                           label_index=None,
                           n_col=0):
    """
    Define operations for reading and parsing given files using tensorflow operations.

    :param filename_queue: tensorflow queue of file names
    :param delimiter: delimiter to parse data
    :param data_index: list of feature indices
    :param label_index: label index
    :param n_col: the number of columns to read

    Returns
    -------
    feature and label pipeline
    """
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0.0]] * (n_col-1) + [[0.0]]
    record = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=delimiter)
    if (label_index != None):
        data, label = [record[i] for i in data_index], [record[label_index]]
        return data, label
    else:
        data = [record[i] for i in data_index]
        return data 


def input_pipeline(filenames,
        delimiter,
        batch_size,
        data_index,
        n_col,
        num_epochs=None):
    """
    Creates thread data pipeline on given file names
    
    :param filenames: list of file names to read
    :param delimiter: delimiter to split files
    :param batch_size: size of batch to return for each call
    :param data_index: index of label data
    :param n_col: total number of columns of each row in a file

    Returns
    -------
    feature and label batch pipeline
    """
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    data = create_file_reader_ops(filename_queue,
            delimiter,
            data_index,
            None,
            n_col)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    data_batch = tf.train.shuffle_batch(
            [data],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=4,
            seed=None,
            enqueue_many=False,
            shapes=None,
            allow_smaller_final_batch=True,
            shared_name=None,
            name=None)
    return data_batch


def input_pipeline_one_hot(filenames,
                   delimiter,
                   batch_size,
                   data_index,
                   label_index,
                   n_col,
                   n_out,
                   num_epochs=None):
    """
    Creates thread data pipeline on given file names with one-hot on label

    :param filenames: list of file names to read
    :param delimiter: delimiter to split files
    :param batch_size: size of batch to return for each call
    :param data_index: index of label data
    :param n_col: total number of columns of each row in a file

    Returns
    -------
    feature and one-hot label batch pipeline
    """
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    data, label = create_file_reader_ops(filename_queue,
            delimiter,
            data_index,
            label_index,
            n_col)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    data_batch, label_batch = tf.train.shuffle_batch(
            [data, label],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=4,
            seed=None,
            enqueue_many=False,
            shapes=None,
            allow_smaller_final_batch=True,
            shared_name=None,
            name=None)
    label_one_hot_batch = tf.one_hot(tf.squeeze(tf.cast(label_batch, tf.int32)), n_out)
    return data_batch, label_one_hot_batch
