import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import Utils
import transformer.Constants as Constants
from preprocess.Dataset import get_dataloader
from transformer.Models import Model


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    # def load_data(name, dict_name):
    #     with open(name, 'rb') as f:
    #         data = pickle.load(f, encoding='latin-1')
    #         # print(data.keys())
    #         num_types = data['dim_process']
    #         data = data[dict_name]
    #         return data, int(num_types)
    #
    # print('[Info] Loading train data...')
    # train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    # # print('[Info] Loading dev data...')
    # # dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    # print('[Info] Loading test data...')
    # test_data, _ = load_data(opt.data + 'test.pkl', 'devtest')

    print('[Info] Loading train data...')
    trainloader = get_dataloader(opt.data, opt.batch_size, shuffle=False, domain="train")
    print('[Info] Loading test data...')
    testloader = get_dataloader(opt.data, opt.batch_size, shuffle=False, domain="test")
    with open(opt.data + "adj_mx.pkl", "rb") as f:
        adj_mx = pickle.load(f)
    return trainloader, testloader, opt.num_types, adj_mx


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, weather_info = map(
            lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, weather_info, opt.adj_mx)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        # scale_time_loss = 100
        # loss = event_loss + pred_loss + se / scale_time_loss
        loss = event_loss + pred_loss + se * opt.scaletimeloss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_macro_F1 = 0
    total_micro_F1 = 0
    iter = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, weather_info = map(
                lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time, weather_info, opt.adj_mx)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)
            macro_F1, micro_F1 = Utils.F1(prediction[0], event_type)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            total_macro_F1 += macro_F1.item()
            total_micro_F1 += micro_F1.item()
            iter += 1

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, total_macro_F1 / iter, total_micro_F1 / iter


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt, sw=None):
    """ Start training. """

    last_val_acc = 0.0
    PATH = None
    if opt.save_model:
        with open("./models/args_" + str(opt.tag) + ".pkl", "wb") as f:
            pickle.dump(opt, f)
        torch.save(model, "./models/base_" + str(opt.tag) + ".pkl")
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    valid_macro_F1 = []
    valid_micro_F1 = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time, valid_macro, valid_micro = eval_epoch(model, validation_data,
                                                                                   pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, macro F1: {macro_f1: 8.5f}, '
              'micro F1: {micro_f1: 8.5f}, '
              'elapse: {elapse: 3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, macro_f1=valid_macro,
                      micro_f1=valid_micro, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        valid_macro_F1 += [valid_macro]
        valid_micro_F1 += [valid_micro]

        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}, Maximum macro F1: {macro_f1: 8.5f},'
              'Maximum macro F1: {micro_f1: 8.5f},'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse),
                      macro_f1=max(valid_macro_F1),
                      micro_f1=max(valid_micro_F1)))

        if opt.record:
            sw.add_scalar('loglikelihood/train', train_event, global_step=epoch_i)
            sw.add_scalar('accuracy/train', train_type, global_step=epoch_i)
            sw.add_scalar('RMSE/train', train_time, global_step=epoch_i)
            sw.add_scalar('loglikelihood/valid', valid_event, global_step=epoch_i)
            sw.add_scalar('accuracy/valid', valid_type, global_step=epoch_i)
            sw.add_scalar('RMSE/valid', valid_time, global_step=epoch_i)
            sw.add_scalar('Macro F1/valid', valid_macro, global_step=epoch_i)
            sw.add_scalar('Micro F1/valid', valid_micro, global_step=epoch_i)

            # logging
            with open(opt.log, 'a') as f:
                f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {macro_f1: 8.5f}, {micro_f1: 8.5f}\n'
                        .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time, macro_f1=valid_macro,
                                micro_f1=valid_micro))

        if opt.save_model:
            if valid_type > last_val_acc:
                print("val acc:" + str(float(valid_type)) + " > last val acc:" + str(float(last_val_acc)))
                if PATH is not None:
                    os.remove(PATH)
                    print("Remove:" + PATH)
                last_val_acc = valid_type
                PATH = "./models/" + opt.tag + "_" + str(epoch_i) + ".pth"
                torch.save(model.state_dict(), PATH)
                print("Save model as:" + PATH)

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="data/data_qx_space/")

    # parser.add_argument('-data', type=str, default="data/data_so/fold1/")
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_value', type=int, default=6)
    parser.add_argument('--d_inner_hid', type=int, default=128)
    parser.add_argument('--d_k', type=int, default=16)
    parser.add_argument('--d_v', type=int, default=16)
    parser.add_argument('--num_types', type=int, default=10)

    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--scaletimeloss', type=float, default=1e-3)

    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    # parser.add_argument('--save_model', action="store_true")

    parser.add_argument('--log', type=str, default='./logs/baseline.txt')
    parser.add_argument('--tag', type=str, default='baseline')
    parser.add_argument("--dev", type=str, default="cpu")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--save_model', action="store_true")

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device(opt.dev)

    if opt.record:
        sw = SummaryWriter(comment=opt.log)

        # setup the log file
        with open(opt.log, 'w') as f:
            f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types, opt.adj_mx = prepare_dataloader(opt)
    opt.adj_mx = torch.Tensor(opt.adj_mx).to(opt.device)

    """ prepare model """
    model = Model(
        num_types=num_types,
        d_model=opt.d_model,
        enc_dim=opt.d_value,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        device=opt.device
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    if opt.record:
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt, sw)
    else:
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()

