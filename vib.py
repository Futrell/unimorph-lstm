import sys
import math
import random

import rfutils
import torch

LOG2 = math.log(2)

class GaussianVIB(torch.nn.Module):
    def __init__(self, encoder, decoder, K):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.K = K

    def forward(self, x):
        statistics = self.encoder(x)
        mu = statistics[:self.K]
        std = torch.nn.functional.softplus(statistics[self.K:], beta=1) # enforce positive-semidefinite covariance matrix
        z = reparameterize(mu, std)
        y = self.decoder(z)
        return (mu, std), y

def reparameterize(mu, std):
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + eps * std

class FFEncoder(torch.nn.Module):
    def __init__(self, structure):
        super().__init__()
        def layers():
            for a, b in rfutils.sliding(structure, 2):
                yield torch.nn.Linear(a, b)
                yield torch.nn.ReLU()
        self.ff = torch.nn.Sequential(*layers())

    def forward(self, x):
        return self.ff(x)

def one_hot(k, n):
    y = torch.zeros(n)
    y[k] = 1
    return y

def logistic(x):
    return 1 / (1 + (-x).exp())

def example(num_bits=2, beta=1/10, num_epochs=10000, print_every=100, **kwds):
    # X: 00, 01, 10, 11.
    # Y: 0, 0, 1, 1
    K = 2**num_bits
    model = GaussianVIB(FFEncoder([K, 10, 5*2]), torch.nn.Linear(5, 1), 5)
    opt = torch.optim.Adam(params=model.parameters(), **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        index = random.choice(range(K))
        y = (1+index) > K/2
        x = one_hot(index, K)
        (mu, std), y_hat_logit = model(x)
        y_hat = logistic(y_hat_logit)
        yz_loss = -(y * y_hat.log() + (1-y) * (1-y_hat).log())
        xz_loss = -(1/2)*(1 + 2*std.log() - mu**2 - std**2).mean()
        loss = yz_loss + beta*xz_loss
        loss.backward()
        opt.step()
        if i % print_every == 0:
            print("H[Y|Z] =", yz_loss.item() / LOG2, "I[X:Z] =", xz_loss.item() / LOG2)
    return model

if __name__ == '__main__':
    example()
            
        
    
        
