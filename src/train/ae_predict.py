"""
File: train/ae_predict.py
 - Contain source code for reconstruction error calculation of autoencoder model.
"""
import sys
sys.path.append("..")

import argparse
import json
import numpy as np

from model.auto_encoder import Autoencoder
from model.common import load_data_no_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=100, type=int, help='mini batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='optimizer')
    parser.add_argument('--epoch', default='200', type=int, help='train epoch')
    parser.add_argument('--reg_param', default=0.01, type=float, help='the regularization parameter for sparsity')

    parser.add_argument('--input', default=28, type=int, help='input dimension')
    parser.add_argument('--z', default=5, type=int, help='z dimension')
    parser.add_argument('--hidden', default=[14, 7], type=int, help='the number of units of each hidden layer', nargs='+')

    parser.add_argument('--train', default='', type=str, help='train data path', nargs='+')
    parser.add_argument('--train_n', default='', type=int, help='number of train data')
    parser.add_argument('--test', default='', type=str, help='test data path', nargs='+')
    parser.add_argument('--data_index', default='', type=int, help='data index', nargs='+')
    parser.add_argument('--display_step', default='', type=int, help='display step')
    parser.add_argument('--early_stop', default='', type=int, help='maximum step for early stopping')

    parser.add_argument('--log_dir', default="result", type=str, help='save path')
    parser.add_argument('--device', default="gpu", type=str, help='device')
    parser.add_argument('--use_pretrained', default=False, action='store_true')
    parser.add_argument('--pretrain_path', default='', type=str, help='path of pretrained model')

    # Parse args
    args = parser.parse_args()
    # Global args
    batch_size = args.batch
    lr = args.lr
    optimizer_type = args.optimizer
    epoch = args.epoch
    hidden = args.hidden
    reg_param = args.reg_param

    train = args.train
    train_n = args.train_n
    test = args.test
    data_index = args.data_index

    _in = args.input
    z = args.z
    log_dir = args.log_dir
    display_step = args.display_step
    early_stop = args.early_stop
    device = args.device

    use_pretrained = args.use_pretrained
    pretrain_path = args.pretrain_path

    hyper_str = vars(args)
    hyper = open(log_dir + '/hyperparameter', 'w')
    hyper.write(json.dumps(hyper_str))
    hyper.close()

    # Source unlabeled autoencoder
    network_architecture = dict(hidden=hidden, n_input=_in, n_z=z) 
    ae = Autoencoder(
            network_architecture,
            lr,
            optimizer_type,
            batch_size,
            train,
            data_index,
            train_n,
            log_dir,
            epoch,
            reg_param,
            use_pretrained=use_pretrained,
            pretrain_path=pretrain_path)
    test_d = load_data_no_out(test, ',', data_index)
    train_d = load_data_no_out(train, ',', data_index)

    ae.init_session()
    train_reconst = ae.predict_decoder(train_d)
    test_reconst = ae.predict_decoder(test_d)

    train_error = np.linalg.norm(train_reconst - train_d, axis=1)
    test_error = np.linalg.norm(test_reconst - test_d, axis=1)
    train_rel_error = train_error / np.linalg.norm(train_d, axis=1)
    test_rel_error = test_error / np.linalg.norm(test_d, axis=1)

    np.savetxt(log_dir + "/train_error", train_error)
    np.savetxt(log_dir + "/test_error", test_error)
    np.savetxt(log_dir + "/train_rel_error", train_rel_error)
    np.savetxt(log_dir + "/test_rel_error", test_rel_error)

    with open(log_dir + "/err_summary", 'w') as f:
        f.write("train_error_mean: %.9f\n" % np.average(train_error))
        f.write("test_error_mean: %.9f\n" % np.average(test_error))
        f.write("train_rel_error_mean: %.9f\n" % np.average(train_rel_error))
        f.write("test_rel_error_mean: %.9f\n" % np.average(test_rel_error))

