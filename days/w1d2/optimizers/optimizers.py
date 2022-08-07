"""In this script, we implement 3 optimizers. We will optimize https://en.wikipedia.org/wiki/Rosenbrock_function which is a benchmark problem in optimization.

Read this article:
https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c

Warning: Notations in the article are not the same as here. Try to forget about the article when doing this TP. Just infer what has to be done from the __init__ and the questions.

Then try to implement them here.

Some details are omitted, there is little chance that you will pass the tests which will compare your implementation with Pytorch's implementation, but that's ok. So read the solution after 5 minutes of try & error

Bonus for maths people:
- https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
- https://optimization.cbe.cornell.edu/index.php?title=AdaGrad

"""

import torch
from typing import Tuple
import optimizers_tests as tests


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
SGD:
0. Read the _opt_rosenbrock code.
1. Why do we have to zero the gradient in PyTorch? Why do we use the word 'stochastic' in 'Stochastic gradient descent' in the context of deep learning?
2. Below, read the method zero_grad. You can note that in PyTorch, to zero a gradient means assigning None.
3. Implement step.
    - Why do we need self.b in the __init__?
    - weight_decay: What is the formula of the update when there is some weight_decay (aka we penalize each parameter squared)? Assume wd absorbs the constant.
        Tip: let's say we optimize L(X, y) = (ax_1 + bx_2 + c - y)^2 with respect to a, b and c.
        Adding weight_decay means that instead of minimizing L, we minimize g(X, y) =  L(X, y) + wd(a^2 + b^2 + c^2)/2
        For this example, what is the formula of the gradient wrt a,b and c?
    3.2 Separate the cases self.momentum equals zero or not.
    3.3 Why do we need the 'with torch.no_grad()' context manager?
4. There are multiple ways to implement SGD, so don't panic if there is some ambiguity and look at the solution to compare with the PyTorch implementation when you have used every variable.
"""


class _SGD:
    def __init__(
        self, params, lr: float, momentum: float, dampening: float, weight_decay: float
    ):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.momentum = momentum  # here, it's the correct definition of momentum
        self.dampening = dampening  # Tip: replace 1-momentum by 1-dampening
        self.b = [None for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                ...


"""
Bonus:
_RMSprop: Using the square of the gradient to adapt the lr
- What is the formula of the update when there is some weight_decay? Assume wd absorbs the constant.
- Update the squared gradient.
- Why do we use the gradient squared? Why do we say that the lr in _RMSprop is adaptive?
- eps should be outside the squared root. How would you adapt eps if it were inside?
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

        self.b2 = [torch.zeros_like(p) for p in self.params]  # gradient squared estimate
        self.b = [torch.zeros_like(p) for p in self.params]  # gradient estimate

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                ...


"""
Bonus: Adam, by far the most used optimizer.
It's a combination of SGD + RMSProps and uses one momentum for the gradient, and another for the gradient squared.
- update b1, b2 
- compute b1_hat, b2_hat    
    - b1_hat = self.b1[i] / (1.0 - self.beta1**self.t)
    - same for b2_hat
    - Why this formula?
- update the gradient
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
                ...


"""
Bonus: 
- Give a reason to use SGD instead of Adam.
- What is an abstract class in python?
- Modify the script to use an abstract class.
"""

if __name__ == "__main__":
    tests.test_sgd(_SGD)
    tests.test_rmsprop(_RMSprop)
    tests.test_adam(_Adam)
