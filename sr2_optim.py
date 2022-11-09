import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
import numpy as np
import logging


class SR2optim(Optimizer):
    """Implementation of SR2 algorithm
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g3=0.5, lmbda=0.001, sigma=0.75, weight_decay=0.2,
                 beta=0.9):

        if not 0.0 <= nu1 <= nu2 < 1.0:
            raise ValueError("Invalid parameter: 0 <= {} <= {} < 1".format(nu1, nu2))
        if not g1 > 1.0:
            raise ValueError("Invalid g1 parameter: {}".format(g1))
        if not 0 < g3 <= 1:
            raise ValueError("Invalid g3 value: {}".format(g3))

        self.sigma = sigma
        self.successful_steps = 0
        self.failed_steps = 0
        self.stop_counter = 0
        self.beta = beta
        self.norm_s = 0
        self.denom = None
        self.current_params = []

        logging.basicConfig(level=logging.INFO)
        defaults = dict(nu1=nu1, nu2=nu2, g1=g1, g3=g3, lmbda=lmbda, sigma=sigma, weight_decay=weight_decay)
        super(SR2optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SR2optim, self).__setstate__(state)

    def _copy_params(self):
        self.current_params = []
        for param in self.param_groups[0]['params']:
            self.current_params.append(deepcopy(param.data))


    def _load_params(self, current_params):
        i = 0
        for param in self.param_groups[0]['params']:
            param.data[:] = current_params[i]
            i += 1

    def get_step(self, x, grad, sigma, lmbda):
        raise NotImplementedError

    def get_denom(self, i, sigma, grad, precond):
        return self.sigma
    
    def additional_initializations(self):
        self.denom = 0

    def cumulate_elements(self, i, s_data, flat_s_data, denom):
        pass
    
    def update_precond(self):
        pass

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # load parameters
        group = self.param_groups[0]

        loss = None
        h_x = None
        if closure is not None:
            loss, h_x = closure()

        loss.backward()
        f_x = loss.item()
        h_x *= group['lmbda']
        current_obj = f_x + h_x
        l = h_x

        # saving the parameters in case the step is rejected
        self._copy_params()

        self.norm_s = 0
        phi_x = f_x
        gts = 0
        stop = False
        do_updates = True
        i = 0

        self.additional_initializations()
        
        for x in group['params']:
            if x.grad is None:
                continue

            # Perform weight-decay
            x.data.mul_(1 - 0.001 * group['weight_decay'])

            grad = x.grad.data

            state = self.state[x]
            if len(state) == 0:
                state['s'] = torch.zeros_like(x.data)
                state['vt'] = torch.zeros_like(grad)
                state['precond'] = torch.zeros_like(x.data)

            # Direction with momentum
            state['vt'].mul_(self.beta).add_(1 - self.beta, grad)
            flat_v = state['vt'].view(-1)

            # get denominator
            self.get_denom(i, self.sigma, grad, state['precond'])

            # Compute the step s
            state['s'].data = self.get_step(x, state['vt'], self.denom, group['lmbda'])  # replace sigma with denom
            self.norm_s += torch.sum(torch.square(state['s'])).item()

            # phi(x+s) ~= f(x) + v^T * s
            flat_s = state['s'].view(-1)
            gts += torch.dot(flat_v, flat_s).item()
            
            # Some versions of SR2 need additional elements
            self.cumulate_elements(i, state['s'].data, flat_s.data, self.denom)

            # Update the weights
            x.data = x.data.add_(state['s'].data)
            i += 1

        phi_x += gts

        # f(x+s), h(x+s)
        fxs, hxs = closure()
        hxs *= group['lmbda']

        # Rho
        delta_model = current_obj - (phi_x + hxs)

        if delta_model < -1e-4:
            logging.error('denominator is negatif {} '.format(delta_model))
            logging.info('current_objectif = {} '.format(current_obj))
            logging.info('phi = {} '.format(phi_x))
            logging.info('h(x+s) = {} '.format(hxs))
            stop = True
            do_updates = False
            rho = np.NAN

        elif -1e-4 <= delta_model <= 0:
            rho = 0
            self.stop_counter += 1
            logging.info('denominator of rho is slightly negatif  {} '.format(delta_model))
            do_updates = False
        else:
            rho = (current_obj - fxs - hxs) / delta_model
            self.stop_counter = 0

        if self.stop_counter > 30:
            stop = True

        # Updates
        if do_updates:
            if rho >= self.param_groups[0]['nu1']:
                logging.debug('step accepted')
                loss = fxs
                l = hxs
                loss.backward()
                self.successful_steps += 1
                
                self.update_precond()
            else:
                # Reject the step
                logging.debug('step rejected')
                self._load_params(self.current_params)
                self.sigma *= group['g1']
                self.failed_steps += 1

            if rho >= self.param_groups[0]['nu2']:
                self.sigma *= group['g3']

        return loss, l, self.norm_s, self.sigma, rho, stop


class SR2optiml1(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, grad, denom, lmbda):
        step = torch.max(x.data - grad / denom - (lmbda / denom), torch.zeros_like(x.data)) - \
               torch.max(-x.data + grad / denom - (lmbda / denom), torch.zeros_like(x.data)) - x.data
        return step


class SR2optiml0(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, grad, denom, lmbda):
        g_over_denom = grad / denom
        if torch.is_tensor(denom):
            step = torch.where(torch.abs(x.data - g_over_denom) >= torch.sqrt(2 * lmbda / denom),
                           -g_over_denom, -x.data)
        else:
            step = torch.where(torch.abs(x.data - g_over_denom) >= np.sqrt(2 * lmbda / denom),
                               -g_over_denom, -x.data)
        return step
    

class SR2optiml12(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, vt, denom, lmbda):
        X = x.data - vt / denom
        p = 54 ** (1 / 3) / 4 * (2 * lmbda / denom) ** (2 / 3)
        a = torch.abs(X)
        phi = torch.arccos(lmbda / (4 * denom) * (a / 3) ** (-3 / 2))
        s = 2 / 3 * a * (1 + torch.cos(2 * torch.pi / 3 - 2 / 3 * phi))

        step = torch.where(X > p, s - x.data, torch.where(X < -p, -s - x.data, -x.data))
        return step

    

class SR2optiml23(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def get_step(self, x, grad, denom, lmbda):
        g_over_denom = grad / denom
        X = x.data - g_over_denom
        L = 2 * lmbda / denom
        phi = torch.arccosh(27 / 16 * (X ** 2) * (L ** (-3 / 2)))
        A = 2 / np.sqrt(3) * L ** (1 / 4) * (torch.cosh(phi / 3)) ** (1 / 2)
        cond = 2 / 3 * (3 * L ** 3) ** (1 / 4)
        s = ((A + ((2 * torch.abs(X)) / A - A ** 2) ** (1 / 2)) / 2) ** 3

        step = torch.where(X > cond, s - x.data, torch.where(X < -cond, -s - x.data, -x.data))
        return step
    
    
class SR2optimAdam(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denom = []
        
    def get_denom(self, i, sigma, grad, precond):
        self.denom = precond.mul_(0.9).addcmul_(1 - 0.9, grad, grad)          # exponential moving average precond
        self.denom.sqrt() / (1 + 1e-6)   # sqrt had bias_correction 2
        self.denom.add_(sigma)
        
    def additional_initializations(self):
        for x in self.param_groups[0]['params']:
            if x.grad is None:
                continue

            state = self.state[x]
            if len(state) == 0:
                self.denom = torch.zeros_like(x.data)
                
                
class SR2optimAdaml0(SR2optimAdam, SR2optiml0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SR2optimAdaml1(SR2optimAdam, SR2optiml1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SR2optimAdaml12(SR2optimAdam, SR2optiml12):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SR2optimAdaml23(SR2optimAdam, SR2optiml23):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SR2optimAndrei(SR2optim):

    def __init__(self, *args, **kwargs):
        self.A = []
        self.B = []
        self.current_grads = []
        self.trA2 = 0
        self.norm_s_sq = 0
        self.sT_B_s = 0
        super().__init__(*args, **kwargs)
        self.initialize_A_B()

    def _copy_params(self):
        self.current_params = []
        self.current_grads = []
        for param in self.param_groups[0]['params']:
            self.current_params.append(deepcopy(param.data))
            self.current_grads.append(deepcopy(param.grad.data))

    def get_sTy(self):
        sTy = 0
        j = 0
        for param in self.param_groups[0]['params']:
            s = param.data - self.current_params[j].data
            y = param.grad.data - self.current_grads[j].data
            flat_s = s.view(-1)
            flat_y = y.view(-1)
            sTy += torch.dot(flat_s, flat_y).item()
            j += 1
        return sTy

    def initialize_A_B(self):
        for param in self.param_groups[0]['params']:
            self.A.append(torch.ones_like(param.data))
            self.B.append(torch.ones_like(param.data))

    def get_denom(self, i, sigma, grad, precond):
        mask = self.B[i].data > 1e-5
        return self.B[i].data * mask.data + sigma
  
    def additional_initializations(self):
        self.trA2 = 0
        self.norm_s_sq = 0
        self.sT_B_s = 0
        
    def cumulate_elements(self, i, s_data, flat_s_data, denom):
        self.A[i] = torch.pow(s_data, 2)
        self.trA2 += torch.sum(torch.pow(s_data, 4))
        self.norm_s_sq += self.norm_s ** 2
        self.sT_B_s += torch.dot(flat_s_data, torch.mul(denom.view(-1), flat_s_data)).item()
        
    def update_precond(self):
        sT_y = self.get_sTy()
        q = (sT_y + self.norm_s_sq - self.sT_B_s) / self.trA2
        
        # update B := B - I + q*A
        k = 0
        for param in self.param_groups[0]['params']:
            self.B[k] = torch.add(torch.add(self.B[k], -1), self.A[k], alpha=q, out=self.B[k])
            k += 1
            

class SR2optimAndreil0(SR2optimAndrei, SR2optiml0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class SR2optimAndreil1(SR2optimAndrei, SR2optiml1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

class SR2optimAndreil12(SR2optimAndrei, SR2optiml12):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class SR2optimAndreil23(SR2optimAndrei, SR2optiml23):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


