import torch
import numpy as np


class Adam(object):
    # For stationary kernel
    def __init__(self, lengthscale, variance, noise, clamp_noise=None, init_lr=0.5, betas=(0.9, 0.99), opt_lengthscale=True, opt_variance=True, opt_relativenoise=True):
        # initialize optimier
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.opt_iter = 0
        self.clamp_noise = clamp_noise
        self.loglengthscale = torch.nn.Parameter(torch.tensor(np.log(lengthscale), requires_grad=True))
        self.logvariance = torch.nn.Parameter(torch.tensor(np.log(variance), requires_grad=True))
        self.logrelativenoise = torch.nn.Parameter(torch.tensor(np.log(noise / variance), requires_grad=True))
        self.opt_lengthscale = opt_lengthscale
        self.opt_variance = opt_variance
        self.opt_noise = opt_relativenoise
        self.betas = betas
        self.update_parameters()
        self.set_lr(init_lr)

    def update_parameters(self):
        self.lengthscale_torch = torch.exp(self.loglengthscale)
        self.variance_torch = torch.exp(self.logvariance)
        if self.clamp_noise is not None:
            self.noise_torch = torch.clamp(torch.exp(self.logrelativenoise), min=self.clamp_noise) * self.variance_torch
        else:
            self.noise_torch = torch.exp(self.logrelativenoise) * self.variance_torch

    def set_lr(self, lr):
        self.learning_rate = lr
        self.history_parameters = []
        self.history_grads = []
        self.parameters = []
        if self.opt_lengthscale:
            self.parameters.append(self.loglengthscale)
        if self.opt_variance:
            self.parameters.append(self.logvariance)
        if self.opt_noise:
            self.parameters.append(self.logrelativenoise)
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate, betas=self.betas)

    def step(self, grad_lengthscale, grad_variance, grad_noise):
        self.optimizer.zero_grad()
        self.lengthscale_torch.backward(gradient=torch.tensor(grad_lengthscale))
        self.variance_torch.backward(gradient=torch.tensor(grad_variance), retain_graph=True)
        self.noise_torch.backward(gradient=torch.tensor(grad_noise))
        self.optimizer.step()
        self.update_parameters()
        self.history_parameters.append(np.array(list(map(lambda x: x.item(), self.parameters))))
        self.history_grads.append(np.array(list(map(lambda x: x.grad.item(), self.parameters))))
        self.opt_iter += 1
        self.lengthscale = self.lengthscale_torch.item()
        self.variance = self.variance_torch.item()
        self.noise = self.noise_torch.item()
