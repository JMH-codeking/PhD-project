import numpy as np
from utils import *
from signal_gen import *
import torch
import torch.nn as nn
from model import *
# use initilisation of some commands
symbol_num = 100000
d_obj = 2
n_heads = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
y, label = multi_cell_gen(M = 2, cell_num = 3, SNR = 6, K = symbol_num)

# separate the real and complex parts for _y
y = torch.tensor(y, dtype = torch.cfloat)
y = torch.view_as_real(y)

label = torch.tensor(
    label,
    dtype = torch.long
)

y_train, label_train = y[:, 0:50000], label[:, 0:50000]
y_test, label_test = y[:, 50000:], label[:, 50000:]
print (f'training data shape: {y_train.shape}')
print (f'training label shape: {label_train.shape}')
print (f'testing data shape: {y_test.shape}')
print (f'testing label shape: {label_test.shape}')


loss = nn.CrossEntropyLoss()
import hiddenlayer as hl

canvasl = hl.Canvas()
historyl = hl.History()
window_size = 20 
train_cnt = 0
acc_train_list = list()
acc_train_lstm_list = list()

def norm(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

import time
from matplotlib import pyplot as plt
canvasl = hl.Canvas()
historyl = hl.History()
input_size = 2 # input size for each LSTM cell (complex numbers)
hidden_size = 20 # hidden state size for LSTM cell
num_classes = 4 # number of classes for classification
window_size = 20
acc_STransformer = list()
acc_STransformer_1 = list()
acc_LSTM = list()
acc_LSTM_1 = list()
training_time =list()
ser_ST_1 = 0
ser_ST_15 = 0
gpu_num = torch.cuda.device_count()

ser_SB_1 = 0
ser_SB_15 = 0
idx_list = list()
for idx in range (1):
    start = time.time()
    train_transformer_1 = True

    enc_transformer_1= Stransformers(d_obj, n_heads)

    enc_transformer_1 = torch.nn.DataParallel(
        enc_transformer_1, 
        device_ids=list(range(gpu_num))
    ).to(device)
    _x = y_train[idx]
    _y= label_train[idx]

    __x = y_test[idx]
    __y = label_test[idx]

    Loss = nn.CrossEntropyLoss()
    optimiser_transformer_0 = torch.optim.Adam(
        enc_transformer_1.parameters(),
        lr = 1e-4,
    )


    cnt_print_transformer_1 = 0

    for epoch in range (10):
        acc_100_trans = []
        acc_trans = 0
        __cnt = 0
        print (f'-------- {epoch} --------')
        for cnt in range (0, symbol_num - window_size + window_size, window_size):
            _cnt = cnt + window_size
            x = _x[cnt: cnt + window_size].to(device)
            x = norm(x)
            y = _y[cnt: cnt + window_size].to(device)

            # carrier 0
            # if train_transformer_1:
            Y_transformer = enc_transformer_1(x.permute(1, 0, 2))
            _, predicted = torch.max(Y_transformer, 1)
            acc_transformer_1 = (predicted == y).sum().item() / window_size 
            ser_ST_1 = 1 - acc_transformer_1
            optimiser_transformer_0.zero_grad()
            loss = Loss(Y_transformer, y)
            loss.backward()
            optimiser_transformer_0.step()
            # carrier 1 training or not
            acc_100_trans.append(acc_transformer_1)
            acc_trans += acc_transformer_1
            __cnt += 1


            if acc_transformer_1 >= 0.9:
                for param_group in optimiser_transformer_0.param_groups:
                    param_group['lr'] = 1e-5
                if len(acc_100_trans) == 5 and train_transformer_1 == True: 
                    if all(number >= 0.95 for number in acc_100_trans):
                        # torch.save(enc_transformer_1.state_dict(), './model/model_state_dict/STransformer_2dB.pt')
                        # end = time.time()
                        # _training = end - start
                        # training_time.append(_training)
                        acc_STransformer.append(_cnt)
                        print (f'STransformer: {_cnt}')
                        break
                        # idx_list.append(idx)
                        # train_transformer_1 = False
                    acc_100_trans.pop(0)
            # if train_transformer_1:
            #     optimiser_transformer_0.zero_grad()
            #     loss = Loss(Y_transformer, y)
            #     loss.backward()
            #     optimiser_transformer_0.step()
            # else:
            #     ser_ST_1 = None 
        
            historyl.log(
                cnt,
                STransformer_carrier_1 = ser_ST_1,
                STransformer_carrier_15 = ser_ST_15,

                SBRNN_carrier_1 = ser_SB_1,
                SBRNN_carrier_15 = ser_SB_15,
            )
            # canvasl.draw_plot(
            #     [
            #         historyl['STransformer_carrier_1'],
            #     ],
            #     xlabel = 'Window Step',
            #     ylabel = 'SER',
            #     _title = 'STransformer Learning',
            #     _semilogy = True
            # )


    acc_test = 0
    cnt_test = 0
    acc_total = []
    for cnt in range (len(y_test) - window_size + 1):

        __x = norm(__x)
        x = __x[cnt: cnt + window_size].to(device)
        y = __y[cnt: cnt + window_size].to(device)

        # carrier 0
        Y_transformer = enc_transformer_1(x.permute(1, 0, 2))
        _, predicted = torch.max(Y_transformer, 1)
        acc_cnt_test = (predicted == y).sum().item() / window_size
        acc_test += acc_cnt_test

        cnt_test += 1
        if cnt % 1000 == 0:
            print (f'------\n{cnt}\n-------')
            print (f' -- {acc_test / cnt_test} -- ')
    print (f'\n FINAL for {idx}\n')
    acc_total.append(acc_test / cnt_test)
    print (acc_test / cnt_test) 



            