import torch
import math

class Own_SGD(torch.optim.Optimizer):
    def __init__(self, params, learn_rate = 1e-3, lam = 0, momentum = 0.9):
        defaults = {'learn_rate': learn_rate,'lambda': lam, 'momentum': momentum}
        super().__init__(params,defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['lambda'] != 0:
                    grad = grad + group['lambda'] * p
                state = self.state[p]
                if len(state) == 0:
                    state['v'] = torch.zeros_like(p)
                v = state['v']
                v = group['momentum'] * v - group['learn_rate'] * grad
                state['v'] = v
                with torch.no_grad():
                    p += v

class Own_RAdam(torch.optim.Optimizer):
    def __init__(self, params, learn_rate = 1e-3, betas = (0.9, 0.999), eps = 1e-8, lam = 0):
        defaults = {'learn_rate': learn_rate, 'betas': betas, 'eps': eps, 'lambda': lam}
        self.rho_inf = 2 / (1 - betas[1]) - 1
        super().__init__(params,defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # past gradients
                    state['m'] = torch.zeros_like(p)
                    # past squared gradients
                    state['v'] = torch.zeros_like(p)
                state['step'] +=1
                grad = p.grad
                # The L_2 regularization
                if group['lambda'] != 0:
                    grad = grad + group['lambda'] * p
                #R
                [B1, B2] = group['betas']
                den = (1 - B2 ** state['step'])
                rho = self.rho_inf - (2 * state['step'] * B2 ** state['step']) / den
                if rho <= 4:
                    with torch.no_grad():
                        p -= group['learn_rate'] * grad
                    continue
                num = (rho -4) * (rho - 2) * self.rho_inf
                den = (self.rho_inf - 4) * (self.rho_inf - 2) * rho
                rt = math.sqrt(num / den)
                # Get the weights and past weights exp-decay average
                m, v = state['m'], state['v']
                m = B1 * m + (1 - B1) * grad
                v = B2 * v + (1 - B2) * grad.pow(2)
                # Update the averages
                state['m'] = m
                state['v'] = v
                # Get the terms to compute the grad update
                mh = m / (1 - B1 ** state['step'])
                vh = v / (1 - B2 ** state['step'])
                # Update the grads
                with torch.no_grad():
                    p -= group['learn_rate'] * mh /(torch.sqrt(vh) + group['eps']) * rt

