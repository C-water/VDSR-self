from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from model import Net
from torch.autograd import Variable
from dataset import get_training_set , get_test_set

torch.cuda.set_device(0)  # use the chosen gpu

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train(epoch):

    adjust_learning_rate(optimizer, epoch-1)

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    #model.train()

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
        optimizer.step()

        #print("===> Epoch({}/{}): Loss: {:.4f}".format(epoch, len(training_data_loader), loss.item()))
    epoch_loss = epoch_loss / len(training_data_loader)
    #Loss_list.append(epoch_loss)
    print("===> Epoch {} Complete: Avg. Loss: {:.10f}".format(epoch, epoch_loss))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            mse = criterion(model(inputs), targets)
            calc_psnr = 10 * log10(1 / mse.item())
            avg_psnr += calc_psnr

    avg_psnr = avg_psnr /len(testing_data_loader)
    #Psnr_list.append(avg_psnr)
    print("===> Avg. PSNR: {:.10f} dB".format(avg_psnr))

def checkpoint(epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# Training settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument('--cuda', default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')
    train_set = get_training_set()
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set,
                                    batch_size=opt.batchSize,
                                    shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set,
                                    batch_size=opt.testBatchSize)

    #Loss_list=[]
    #Psnr_list=[]

    print('===> Building model')
    model = Net().to(device)
    criterion = nn.MSELoss()

    # optimizer = optim.SGD( params=model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.num_epochs+1):
        train(epoch)
        test()
        checkpoint(epoch)


