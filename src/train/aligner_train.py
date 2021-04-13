"""
File: train/aligner_train.py
 - Contain training code for execution for transfer normalization model.
"""
import sys
sys.path.append("..")

import argparse
import json
import os

from model.match_features import Matchnet
from model.common import load_data_no_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=100, type=int, help='mini batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='optimizer')
    parser.add_argument('--epoch', default='5', type=int, help='train epoch')
    parser.add_argument('--reg_layer_idx', default='1', type=int, help='the transfer aliner layer index to apply regularization')
    parser.add_argument('--reg_param', default=0.01, type=float, help='the regularization parameter for sparsity')

    parser.add_argument('--hidden', default=[14, 7], type=int, help='the number of units of each hidden layer', nargs='+')
    parser.add_argument('--encoder1_path', default='', type=str, help='path of encoder 1')
    parser.add_argument('--encoder2_path', default='', type=str, help='path of encoder 2')
    parser.add_argument('--encoder_lr', default=0.01, type=float, help='learning rate of encoder')

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
    reg_layer_idx = args.reg_layer_idx
    reg_param = args.reg_param
    encoder1_path = args.encoder1_path
    encoder2_path = args.encoder2_path
    encoder_lr = args.encoder_lr

    train = args.train
    train_n = args.train_n
    test = args.test
    data_index = args.data_index
    label_index = args.label_index

    log_dir = args.log_dir
    test_step = args.test_step
    early_stop = args.early_stop
    device = args.device

    fix_pretrained = args.fix_pretrained
    use_pretrained = args.use_pretrained
    pretrain_path = args.pretrain_path

    os.makedirs(log_dir, exist_ok=True)
    hyper_str = vars(args)
    hyper = open(log_dir + '/hyperparameter', 'w')
    hyper.write(json.dumps(hyper_str))
    hyper.close()

    # Source unlabeled autoencoder
    network_architecture = dict(
            hidden=hidden)
    matchnet = Matchnet(network_architecture,
            lr,
            optimizer_type,
            batch_size,
            train,
            data_index,
            label_index,
            train_n,
            log_dir,
            epoch,
            encoder1_path=encoder1_path,
            encoder2_path=encoder2_path,
            encoder_lr=encoder_lr,
            reg_layer_idx=reg_layer_idx,
            reg_param=reg_param,
            fix_pretrained=fix_pretrained,
            use_pretrained=use_pretrained,
            pretrain_path=pretrain_path)
    test_d = load_data_no_out(test, ",", data_index)
    matchnet.learn(test_d, test_step, early_stop, device)
