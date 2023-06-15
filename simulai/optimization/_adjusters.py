from typing import List, Dict, Callable
import numpy as np
import torch

from simulai.templates import NetworkTemplate

class WeightsEstimator:

    def __init__(self) -> None:
        pass

    def _clip_grad(self, grad:torch.Tensor=None, size:int=None) -> torch.Tensor:

        if not isinstance(grad, torch.Tensor):
            return torch.zeros(size).detach()
        else:
            return grad.detach()


    def _grad(self, loss:torch.tensor=None, operator:NetworkTemplate=None) -> torch.Tensor:


        if loss.requires_grad:

            loss.backward(retain_graph=True)

            grads = [self._clip_grad(grad=v.grad, size=v.shape) for v in operator.parameters()]

            for v in operator.parameters():
                v.grad = None

            gradients = torch.hstack(
                [
                    v.flatten() for v in grads
                ]
            )
        else:
            gradients = torch.zeros(operator.n_parameters)

        return gradients


##########################################################
# Adjusters designed for equation-based loss terms (PINNs)
##########################################################

class GeometricMean:

    def __init__(self):

        pass

    def __call__(self, residual=List[torch.Tensor],
                       loss_evaluator=Callable,
                       loss_history=Dict[str, float], **kwargs) -> None:

        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        mean_exps = torch.mean(torch.Tensor(exps))
        shifts = [mean_exps - exp for exp in exps]

        weights = [torch.exp(shift).detach() for shift in shifts]

        return weights

class ShiftToMax:

    def __init__(self):

        pass

    def __call__(self, residual=List[torch.Tensor],
                       loss_evaluator=Callable,
                       loss_history=Dict[str, float], **kwargs) -> None:

        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        max_exps = torch.max(torch.Tensor(exps))
        shifts = [max_exps - exp for exp in exps]

        weights = [torch.exp(shift.to(int).detach())/n_res for shift in shifts]

        return weights

########################################################################
# Adjusters designed for balancing overall residual (PINN) contributions
# and data-driven and  initial/boundary conditions
########################################################################

class AnnealingWeights(WeightsEstimator):

    def __init__(self, alpha:float=None, init_weight:float=1.0,
                 bound_weight:float=1.0, extra_data_weight:float=1.0) -> None:

        super().__init__()

        self.alpha = alpha

        self.init_weight = init_weight
        self.bound_weight = bound_weight
        self.extra_data_weight = extra_data_weight

    def _coeff_update(self, loss_ref:torch.tensor=None, loss:torch.tensor=None):

        loss_grad_mean = torch.mean(torch.abs(loss))

        if torch.abs(loss_grad_mean) >= 1e-15:
            coeff_hat = torch.max(torch.abs(loss_ref))/loss_grad_mean
        else:
            coeff_hat = 0

        return coeff_hat

    def __call__(self, residual:torch.tensor=None,
                       operator: NetworkTemplate=None, **kwargs) -> torch.tensor:

        pde = residual[0]
        init = residual[1]
        bound = residual[2]
        extra_data = residual[3]

        pde_grads = self._grad(loss=pde, operator=operator)
        init_grads = self._grad(loss=init, operator=operator)
        bound_grads = self._grad(loss=bound, operator=operator)
        extra_data_grads = self._grad(loss=extra_data, operator=operator)

        init_weight_update = self._coeff_update(loss_ref=pde_grads,
                                                loss=init_grads)
        bound_weight_update = self._coeff_update(loss_ref=pde_grads,
                                                 loss=bound_grads)
        extra_data_weight_update = self._coeff_update(loss_ref=pde_grads,
                                                      loss=extra_data_grads)

        self.init_weight = (self.alpha)*self.init_weight + (1 - self.alpha)*init_weight_update
        self.bound_weight = (self.alpha)*self.bound_weight + (1 - self.alpha)*bound_weight_update
        self.extra_data_weight = (self.alpha)*self.extra_data_weight + (1 - self.alpha)*extra_data_weight_update

        return [1.0, self.init_weight, self.bound_weight, self.extra_data_weight]

class InverseDirichletWeights(WeightsEstimator):

    def __init__(self, alpha:float=None, initial_weights:List[float]=None) -> None:

        super().__init__()

        self.alpha = alpha
        self.default_number = 100

        if not initial_weights:
            self.weights = [1.0]*self.default_number
        else:
            self.weights = initial_weights

    def _coeff_update(self, nominator:torch.tensor=None, loss:torch.tensor=None):

        loss_grad_std = torch.std(loss)

        if torch.abs(loss_grad_std) >= 1e-15:
            coeff_hat = nominator/loss_grad_std
        else:
            coeff_hat = 0

        return coeff_hat

    def __call__(self, residual:List[torch.Tensor]=None,
                       loss_evaluator:Callable=None,
                       loss_history:Dict[str, float]=None,
                       operator:Callable=None, **kwargs) -> None:

        residual_grads = list()

        for res in residual:
            res_loss = res
            residual_grads.append(self._grad(loss=res_loss, operator=operator))

        losses_std = [torch.std(l) for l in residual_grads]

        nominator = torch.max(torch.Tensor(losses_std))

        for j in range(len(residual)):

            weight_update = self._coeff_update(nominator=nominator,
                                               loss=residual_grads[j])

            self.weights[j] = (self.alpha)*self.weights[j] + (1 - self.alpha)*weight_update

        return self.weights[:len(residual)]



