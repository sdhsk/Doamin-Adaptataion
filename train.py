import DaNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm           #进度条库

import data_loader
import mmd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.01
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
DROPOUT = 0.5
N_EPOCH = 1000
BATCH_SIZE = [64, 64]
LAMBDA = 0.25
GAMMA = 10 ^ 3
RESULT_TRAIN = []
RESULT_TEST = []
log_train = open('log_train_a-w.txt', 'w')
log_test = open('log_test_a-w.txt', 'w')

def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA])


def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    batch_j = 0    #inilize
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))  #索引列表
    for batch_id, (data, target) in enumerate(data_src):   #源域
        _, (x_tar, y_target) = list_tar[batch_j]
        data, target = data.data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        x_tar, y_target = x_tar.view(-1, 28 * 28).to(DEVICE), y_target.to(DEVICE)
        model.train()

        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)
        print("x_tar_mmd:",x_tar_mmd.size())
        #定义loss函数
        loss_c = criterion(y_src, target)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = loss_c + LAMBDA * loss_mmd
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录误差
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, batch_id + 1, len(data_src), loss.data)
        print(res_i,'\n')
        batch_j += 1
        if batch_j >= len(list_tar)-1:
            batch_j = 0
    total_loss_train /= len(data_src)
    #计算准确率
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, len(data_src.dataset), acc)
    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def ceshi(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_tar):           #target域
            data, target = data.view(-1,28 * 28).to(DEVICE),target.to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / len(data_tar.dataset)
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, len(data_tar.dataset), accuracy
        )
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    log_test.write(res + '\n')


if __name__ == '__main__':        #直接执行
    #rootdir = '../../../data/office_caltech_10/'
    rootdir='D:/Users/Administrator/Desktop/DANN/data/Original_images/'
    #rootdir='D:/Users/Administrator/Desktop/DANN/data/office_caltech_10/'
    torch.manual_seed(1)   #将随机数生成器的种子设置为固定值
    data_src = data_loader.load_data(
        root_dir=rootdir, domain='dslr', batch_size=BATCH_SIZE[0])
    data_tar = data_loader.load_test(
        root_dir=rootdir, domain='amazon', batch_size=BATCH_SIZE[1])
    model = DaNN.DaNN(n_input=28 * 28, n_hidden=256, n_class=31)
    model = model.to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT)
    for e in tqdm(range(1, N_EPOCH + 1)):
        model = train(model=model, optimizer=optimizer,epoch=e, data_src=data_src, data_tar=data_tar)
        ceshi(model, data_tar, e)
    torch.save(model, 'model_dann.pkl')        #保存模型
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_d_a.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_d_a.csv', res_test, fmt='%.6f', delimiter=',')