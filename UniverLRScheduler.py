import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import math
import argparse
from torchvision.models.vgg import *
import matplotlib.pyplot as plt


class SingleLrscheduler(object):
    def __init__(self, optimizer, setup, minlr=1e-8, warmup='Exp_up', warmup_epoch=10, flat='fix',
                 flat_epoch=100, training='ExponentialLR', metric=None, start_epoch=0, **kwargs):
        super(SingleLrscheduler, self).__init__()
        self.optimizer = optimizer
        self.minlr = minlr
        self.maxlr = args.lr
        self.warmup = warmup
        self.warmup_epoch = warmup_epoch
        self.flat = flat
        self.flat_epoch = flat_epoch
        self.training = training
        self.metric = metric
        self.start_epoch = start_epoch
        self.setup = setup
        self.last_epoch = -1
        assert self.maxlr > self.minlr
        assert self.training in ['LambdaLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
                                 'ReduceLROnPlateau', None]  # only support these now
        assert self.warmup in ['Sin_up', 'Linear_up', 'Exp_up', None]

        if self.training == 'MultiStepLR':
            self.scheduler = MultiStepLR(optimizer=optimizer, milestones=self.setup.MultiStepLR_milestones,
                                         gamma=self.setup.MultiStepLR_gamma)
        elif self.training == 'StepLR':
            self.scheduler = StepLR(optimizer=optimizer, step_size=self.setup.StepLR_step_size,
                                    gamma=self.setup.StepLR_gamma)
        elif self.training == 'ExponentialLR':
            self.scheduler = ExponentialLR(optimizer=optimizer, gamma=self.setup.ExponentialLRepLR_gamma)
        elif self.training == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self.setup.CosineAnnealingLR_T_max,
                                               eta_min=self.setup.CosineAnnealingLR_eta_min)
        elif self.training == 'ReduceLROnPlateauLR':
            self.scheduler = ReduceLROnPlateau(optimizer, mode=self.setup.ReduceLROnPlateauLR_mode,
                                               factor=self.setup.ReduceLROnPlateauLR_factor,
                                               patience=self.setup.ReduceLROnPlateauLR_patience, verbose=True)

        # elif self.training == 'CyclicLR':
        #     self.scheduler = CyclicLR()
        #
        # elif self.training == 'OneCycleLRepLR':
        #     self.scheduler = OneCycleLR()
        #
        # elif self.training == 'CosineAnnealingWarmRestarts':
        #     self.scheduler = CosineAnnealingWarmRestarts()

        if self.training == 'ReduceLROnPlateau':
            assert self.metric is not None

    def step(self, epoch):

        if self.warmup == 'Linear_up' and epoch < self.start_epoch + self.warmup_epoch:
            newlr = (self.maxlr / self.minlr) * (epoch - self.last_epoch) / self.warmup_epoch * self.minlr
            self.adjust_learning_rate(newlr)

        if self.warmup == 'Sin_up' and epoch < self.start_epoch + self.warmup_epoch:
            newlr = (self.maxlr / self.minlr) * math.sin(
                math.pi * ((epoch - self.last_epoch) / self.warmup_epoch / 2)) * self.minlr
            self.adjust_learning_rate(newlr)

        if self.warmup == 'Exp_up' and epoch < self.start_epoch + self.warmup_epoch:
            newlr = self.minlr * (self.maxlr / self.minlr) * 1 / (0.5 ** (epoch - self.warmup_epoch))
            self.adjust_learning_rate(newlr)

        if epoch == self.warmup_epoch:
            self.adjust_learning_rate(self.maxlr)

        if epoch > self.flat_epoch + self.warmup_epoch:
            if self.training == 'ReduceLROnPlateau':
                self.scheduler.step(self.metric)
            else:
                self.scheduler.step()

    def adjust_learning_rate(self, newlr):
        """set new lr"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = newlr

    def get_learning_rate(self):
        """return right now lr"""
        return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('setup record')
    parser.add_argument("--CosineAnnealingLR_T_max", default=30)
    parser.add_argument('--CosineAnnealingLR_eta_min', default=1e-7,
                        help='describle the setting')
    parser.add_argument("--ExponentialLRepLR_gamma", default=0.995)
    parser.add_argument("--lr", default=1e-5)

    args = parser.parse_args()
    model = vgg11()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    SLS = SingleLrscheduler(optimizer=optimizer, setup=args, flat_epoch=20)
    epochs = 500
    x = []
    y = []
    for epoch in range(epochs):
        optimizer.step()
        SLS.step(epoch)
        rlr = SLS.get_learning_rate()
        x.append(epoch)
        y.append(rlr)
    plt.plot(x, y)
    plt.show()
    print('DONE')
