"""
File: train/v1_train.py
 - Contain training code for execution for mlp on top of autoencoder model.
"""
import sys
sys.path.append("..")

import argparse
import json
import os

from model.v1 import V1
from model.common import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=100, type=int, help='mini batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='optimizer')
    parser.add_argument('--epoch', default='5', type=int, help='train epoch')

    parser.add_argument('--input', default=28, type=int, help='input dimension')
    parser.add_argument('--output', default=2, type=int, help='output dimension')
    parser.add_argument('--hidden', default=[14, 7], type=int, help='the number of units of each hidden layer', nargs='+')
    parser.add_argument('--encoder_path', default='', type=str, help='path of autoencoder')
    parser.add_argument('--encoder_lr', default=0.01, type=float, help='learning rate of autoencoder')

    parser.add_argument('--train', default='', type=str, help='train data path', nargs='+')
    parser.add_argument('--train_n', default='', type=int, help='number of train data')
    parser.add_argument('--test', default='', type=str, help='test data path', nargs='+')
    parser.add_argument('--data_index', default='', type=int, help='data index', nargs='+')
    parser.add_argument('--label_index', default='', type=int, help='label index')
    parser.add_argument('--test_step', default=10, type=int, help='test steps to calculate test predictions')
    parser.add_argument('--early_stop', default=40, type=int, help='maximum step for early stopping')

    parser.add_argument('--log_dir', default="result", type=str, help='save path')
    parser.add_argument('--device', default="gpu", type=str, help='device')
    parser.add_argument('--fix_pretrained', default=False, action='store_true')
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
    encoder_path = args.encoder_path
    encoder_lr = args.encoder_lr

    train = args.train
    train_n = args.train_n
    test = args.test
    data_index = args.data_index
    label_index = args.label_index

    _in = args.input
    _out = args.output
    log_dir = args.log_dir
    test_step = args.test_step
    early_stop = args.early_stop
    device = args.device

    use_pretrained = args.use_pretrained
    fix_pretrained = args.fix_pretrained
    pretrain_path = args.pretrain_path

    os.makedirs(log_dir, exist_ok=True)
    hyper_str = vars(args)
    hyper = open(log_dir + '/hyperparameter', 'w')
    hyper.write(json.dumps(hyper_str))
    hyper.close()

    # Source unlabeled autoencoder
    network_architecture = dict(
            hidden=hidden,
            n_input=_in,
            n_output=_out)
    v1 = V1(network_architecture,
            lr,
            optimizer_type,
            batch_size,
            train,
            data_index,
            label_index,
            train_n,
            log_dir,
            epoch,
            encoder_path=encoder_path,
            encoder_lr=encoder_lr,
            use_pretrained=use_pretrained,
            fix_pretrained=fix_pretrained,
            pretrain_path=pretrain_path)
    test_d, test_l = load_data(test, ",", data_index, label_index)
    v1.learn(test_d, test_l, test_step, early_stop, device)
