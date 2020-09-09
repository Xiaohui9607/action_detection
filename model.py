import os
import torch
from torch import nn
import torch.optim as optim
import cv2
import numpy as np
from networks import network, NUM_BEHAVRIORS
from dataloader import build_dataloader
from torch.nn import functional as F


class Model():
    def  __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device

        train_dataloader, valid_dataloader = build_dataloader(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}

        self.net = network(self.opt, self.opt.channels, self.opt.height, self.opt.width)

        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate, weight_decay=1e-4)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        running_loss = 0.0
        for iter_, (images, labels) in enumerate(self.dataloader['train']):
            self.net.zero_grad()
            images = images.permute([0, 1, 2, 3, 4]).unbind(0)
            labels = labels.to(torch.int64)
            labels = labels.view(-1)
            # labels = torch.nn.functional.one_hot(labels.to(torch.int64), NUM_BEHAVRIORS)

            predicted_labels = self.net(images)
            predicted_labels = predicted_labels.view(-1, NUM_BEHAVRIORS)

            loss = self.criterion(predicted_labels, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if iter_ % self.opt.print_interval == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, iter_ + 1, running_loss / len(labels)))
                running_loss = 0.0

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def evaluate(self, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in self.dataloader['valid']:
                images = images.permute([0, 1, 2, 3, 4]).unbind(0)
                predicted_labels = self.net(images)
                _, predicted = torch.max(predicted_labels.data, 2)

                labels = labels.to(torch.int64)
                total += (labels.size(0) * labels.size(1))
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on test images: %d %%' % (
                100 * correct / total))


    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self, path=None):
        if path:
            self.net.load_state_dict(torch.load(path))
        elif self.opt.pretrained_model:
            self.net.load_state_dict(torch.load(self.opt.pretrained_model, map_location=torch.device('cpu')))


if __name__ == "__main__":

    from options import Options

    opt = Options().parse()
    opt.batch_size = 2
    a_model = Model(opt)
    a_model.train()