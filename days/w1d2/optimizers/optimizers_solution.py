"""In this script, we implement 3 optimizers

Rad this article:
https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c

Then try to implement them here.

Some details are omitted, there is little chance that you will pass the tests. So read the solution after 5 minutes of try & error
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
import optimizers_tests as tests


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def _train(model: nn.Module, dataloader: DataLoader, lr, momentum):
    opt = torch.optim.SGD(model.parameters(), lr, momentum)
    for X, y in dataloader:
        opt.zero_grad()
        pred = model(X)
        loss = F.l1_loss(pred, y)
        loss.backward()
        opt.step()
    return model


def _accuracy(model: nn.Module, dataloader: DataLoader):
    n_correct = 0
    n_total = 0
    for X, y in dataloader:
        n_correct += (model(X).argmax(1) == y).sum().item()
        n_total += len(y)
    return n_correct / n_total


def _evaluate(model: nn.Module, dataloader: DataLoader):
    sum_abs = 0.0
    n_elems = 0
    for X, y in dataloader:
        sum_abs += (model(X) - y).abs().sum()
        n_elems += y.shape[0] * y.shape[1]
    return sum_abs / n_elems


def _rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def _opt_rosenbrock(xy, lr, momentum, n_iter):
    w_history = torch.zeros([n_iter + 1, 2])
    w_history[0] = xy.detach()
    opt = torch.optim.SGD([xy], lr=lr, momentum=momentum)

    for i in range(n_iter):
        opt.zero_grad()
        _rosenbrock(xy[0], xy[1]).backward()

        opt.step()
        w_history[i + 1] = xy.detach()
    return w_history


"""
Generalities:
Read the _opt_rosenbrock code.
Why do we have to zero the gradient in PyTorch? 
Why do we use the word 'stochastic' in 'Stochastic gradient descent' in the context of deep learning?

SGD:
Please don't look back at the article. We will try to construct the formula ourself here.
Below, read the method zero_grad. You can note that in PyTorch, to zero a gradient means assigning None.

We will implement successively the pure SGD, the sgd with weight decay, and the SGD with momentum.

Implement steps:
    - # update: Implement the most basic version of SGD possible
    - Why do we need the 'with torch.no_grad()' context manager?
    - # weight_decay: What is the formula of the update when there is some weight_decay (ie when we penalize each parameter squared)? Assume wd absorbs the constant.
        Tip: let's say we optimize L(X, y) = (ax_1 + bx_2 + c - y)^2 with respect to a, b and c.
        Adding weight_decay means that instead of minimizing L, we minimize g(X, y) =  L(X, y) + wd(a^2 + b^2 + c^2)/2
        For this example, what is the formula of the gradient wrt a,b and c?
        In the code, at the beginning of the step function, replace the gradient by g = p.grad + self.wd * p
    - # momentum: Why do we need self.b in the __init__? Add momentum. Separate the cases self.momentum equals zero or not.

There are multiple ways to implement SGD, so don't panic if there is some ambiguity and look at the solution to compare with the PyTorch implementation when you have used every variable.
"""


class _SGD:
    def __init__(
        self, params, lr: float, momentum: float, dampening: float, weight_decay: float
    ):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.momentum = momentum  # here, it's the correct definition of momentum
        self.dampening = dampening  # generally 1-momentum = 1-dampening
        self.b = [None for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None, "Don't forget loss.backward()!"
                # weight decay
                g = p.grad + self.wd * p

                # momentum
                if self.b[i]:
                    self.b[i] = self.momentum * self.b[i] + (1 - self.dampening) * g
                else:
                    self.b[i] = g

                # update
                p -= self.b[i] * self.lr


"""
Adam, by far the most used optimizer.

It's a combination of SGD+RMSProps and uses one momentum for the gradient, and another for the gradient squared.

1. Adam
Tip: There is no if condition in Adam because beta1 and beta2 are always stricly positive.
sum_of_gradient = previous_sum_of_gradient * beta1 + gradient * (1 - beta1) [SGD+Momentum]
sum_of_gradient_squared = previous_sum_of_gradient_squared * beta2 + gradient² * (1- beta2) [RMSProp]
delta = -learning_rate * sum_of_gradient / sqrt(sum_of_gradient_squared)
Update the gradient

2. More stability + regularization
Add self.eps to the denominator, outside the square root.
At the beginning of the step function, replace the gradient by g = p.grad + self.wd * p

3. Adam + Correction
In the __init__, self.b1 and self.b2 are a list of zeros, so we need a correction:
In the update, use a correction: b1_hat = self.b1[i] / (1.0 - self.beta1**self.t)
Generally beta1 = 0.9, and beta2 = 0.999.
Same for b2_hat.
NB: The correction is important only at the beginning of the optimization process (self.t small).
The the correction is needed because self.b1 and self.b2 are initialized at zero.

Try to pass the tests.
"""


class _Adam:
    def __init__(
        self,
        params,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas  # momenti of b1 and b2
        self.eps = eps
        self.wd = weight_decay

        self.b1 = [torch.zeros_like(p) for p in self.params]
        self.b2 = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.b1[i] = self.beta1 * self.b1[i] + (1.0 - self.beta1) * g
                self.b2[i] = self.beta2 * self.b2[i] + (1.0 - self.beta2) * g**2
                b1_hat = self.b1[i] / (1.0 - self.beta1**self.t)
                b2_hat = self.b2[i] / (1.0 - self.beta2**self.t)
                p -= self.lr * b1_hat / (b2_hat.sqrt() + self.eps)


"""
(Bonus) _RMSprop: Using the square of the gradient to adapt the lr
- What is the formula of the update when there is some weight_decay? Assume wd absorbs the constant.
- Update the squared gradient.
- Why do we use the gradient squared? Why do we say that the lr in _RMSprop is adaptive?
- eps should be outside the squared root
- Separate the cases self.momentum zero or not.
"""


class _RMSprop:
    def __init__(
        self,
        params,
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha  # momentum of gradient squared
        self.eps = eps
        self.wd = weight_decay
        self.momentum = momentum

        self.b2 = [torch.zeros_like(p) for p in self.params]  # gradient squared
        self.b = [torch.zeros_like(p) for p in self.params]  # gradient

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.b2[i] = self.alpha * self.b2[i] + (1.0 - self.alpha) * g**2
                if self.momentum:
                    self.b[i] = self.momentum * self.b[i] + g / (
                        self.b2[i].sqrt() + self.eps
                    )
                    p -= self.lr * self.b[i]
                else:
                    p -= self.lr * g / (self.b2[i].sqrt() + self.eps)


"""
Bonus: 
- Give a reason to use SGD instead of Adam.
- What is an abstract class in python?
- Modify the script to use an abstract class.
- What is a Parent class? Modify the script to use a Parent class 
"""

if __name__ == "__main__":
    tests.test_sgd(_SGD)
    tests.test_adam(_Adam)
    tests.test_rmsprop(_RMSprop)
