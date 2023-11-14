from typing import List, Dict, Callable
import numpy as np
import torch

from simulai.templates import NetworkTemplate


class WeightsEstimator:
    def __init__(self) -> None:
        pass

    def _clip_grad(self, grad: torch.Tensor = None, size: int = None) -> torch.Tensor:
        """Filtering gradients.
        Args:
            grad (torch.Tensor): Tensor containing the gradients of all the parameters.
            size (int): Number of parameters.

        Returns:
            torch.Tensor: The clipped gradients. 

        """
        if not isinstance(grad, torch.Tensor):
            return torch.zeros(size).detach()
        else:
            return grad.detach()

    def _grad(
        self, loss: torch.tensor = None, operator: NetworkTemplate = None
    ) -> torch.Tensor:
        """It evaluates the gradients of loss function w.r.t. all the model parameters.

        Args:
            loss (torch.tensor): The current state of the loss function.
            operator (NetworkTemplate): The model being trained. 

        Returns:
            torch.Tensor: The gradients evaluated for all the parameters. 

        """
        if loss.requires_grad:
            loss.backward(retain_graph=True)

            grads = [
                self._clip_grad(grad=v.grad, size=v.shape)
                for v in operator.parameters()
            ]

            for v in operator.parameters():
                v.grad = None

            gradients = torch.hstack([v.flatten() for v in grads])
        else:
            gradients = torch.zeros(operator.n_parameters)

        return gradients


##########################################################
# Adjusters designed for equation-based loss terms (PINNs)
##########################################################


class GeometricMean:
    def __init__(self):
        """Simple and naive approach for balancing the loss terms in which
           they are rescaled to have the same order of magnitude of the geometric 
           mean between all the terms.
        """
        pass

    def __call__(
        self,
        residual=List[torch.Tensor],
        loss_evaluator=Callable,
        loss_history=Dict[str, float],
        **kwargs,
    ) -> None:
        """

        Args:
            residual (List[torch.Tensor]): List containing all the equation-based
                loss terms.
            loss_evaluator (Callable): A Python class or function for evaluating 
                the loss function.

        """
        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        mean_exps = torch.mean(torch.Tensor(exps))
        shifts = [mean_exps - exp for exp in exps]

        weights = [torch.exp(shift).detach() for shift in shifts]

        return weights


class ShiftToMax:
    def __init__(self):
        """Simple and naive approach for balancing the loss terms in which
           they are rescaled to have the same order of magnitude of the maximum value
           of all the terms.
        """

        pass

    def __call__(
        self,
        residual=List[torch.Tensor],
        loss_evaluator=Callable,
        loss_history=Dict[str, float],
        **kwargs,
    ) -> None:
        """

        Args:
            residual (List[torch.Tensor]): List containing all the equation-based
                loss terms.
            loss_evaluator (Callable): A Python class or function for evaluating 
                the loss function.
        """

        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        max_exps = torch.max(torch.Tensor(exps))
        shifts = [max_exps - exp for exp in exps]

        weights = [torch.exp(shift.to(int).detach()) / n_res for shift in shifts]

        return weights


########################################################################
# Adjusters designed for balancing overall residual (PINN) contributions
# and data-driven and  initial/boundary conditions
########################################################################


class AnnealingWeights(WeightsEstimator):
    def __init__(
        self,
        alpha: float = None,
        init_weight: float = 1.0,
        bound_weight: float = 1.0,
        extra_data_weight: float = 1.0,
    ) -> None:
        """Learning rate Annealing technique used 
            for scaling equation-based loss terms (see https://arxiv.org/abs/2001.04536)

        Args:
            alpha (float): 1 - update step.
            init_weight (float): Initial value for the initial condition weight.
            bound_weight (float): Initial value for the boundary condition weight.
            extra_data_weight (float): Initial value for the weight related to 
                data-drive loss terms.

        """
        super().__init__()

        self.alpha = alpha

        self.init_weight = init_weight
        self.bound_weight = bound_weight
        self.extra_data_weight = extra_data_weight

    def _coeff_update(self, loss_ref: torch.tensor = None, loss: torch.tensor = None):
        loss_grad_mean = torch.mean(torch.abs(loss))

        if torch.abs(loss_grad_mean) >= 1e-15:
            coeff_hat = torch.max(torch.abs(loss_ref)) / loss_grad_mean
        else:
            coeff_hat = 0

        return coeff_hat

    def __call__(
        self, residual: torch.tensor = None, operator: NetworkTemplate = None, **kwargs
    ) -> List[torch.tensor]:

        """

        Args:
            residual (torch.tensor): Tensor containing the equation residual.
            operator (NetworkTemplate): Model being trained. 

        Returns:
            List[torch.tensor]: A list containing the updated loss weights.

        """

        pde = residual[0]
        init = residual[1]
        bound = residual[2]
        extra_data = residual[3]

        pde_grads = self._grad(loss=pde, operator=operator)
        init_grads = self._grad(loss=init, operator=operator)
        bound_grads = self._grad(loss=bound, operator=operator)
        extra_data_grads = self._grad(loss=extra_data, operator=operator)

        init_weight_update = self._coeff_update(loss_ref=pde_grads, loss=init_grads)
        bound_weight_update = self._coeff_update(loss_ref=pde_grads, loss=bound_grads)
        extra_data_weight_update = self._coeff_update(
            loss_ref=pde_grads, loss=extra_data_grads
        )

        self.init_weight = (self.alpha) * self.init_weight + (
            1 - self.alpha
        ) * init_weight_update
        self.bound_weight = (self.alpha) * self.bound_weight + (
            1 - self.alpha
        ) * bound_weight_update
        self.extra_data_weight = (self.alpha) * self.extra_data_weight + (
            1 - self.alpha
        ) * extra_data_weight_update

        return [1.0, self.init_weight, self.bound_weight, self.extra_data_weight]


class InverseDirichletWeights(WeightsEstimator):
    def __init__(
        self, alpha: float = None, initial_weights: List[float] = None
    ) -> None:

        """Inverse Dirichlet technique used 
            for scaling equation-based loss terms (see https://iopscience.iop.org/article/10.1088/2632-2153/ac3712/pdf)

        Args:
            alpha (float): 1 - update step.
            initial_weights (List[float]): List containing the initial states of all the loss
                function terms. 
        """

        super().__init__()

        self.alpha = alpha
        self.default_number = 100

        if not initial_weights:
            self.weights = [1.0] * self.default_number
        else:
            self.weights = initial_weights

    def _coeff_update(self, nominator: torch.tensor = None, loss: torch.tensor = None):
        loss_grad_std = torch.std(loss)

        if torch.abs(loss_grad_std) >= 1e-15:
            coeff_hat = nominator / loss_grad_std
        else:
            coeff_hat = 0

        return coeff_hat

    def __call__(
        self,
        residual: List[torch.Tensor] = None,
        loss_evaluator: Callable = None,
        loss_history: Dict[str, float] = None,
        operator: Callable = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """

        Args:
            residual (List[torch.Tensor]): List of equation-based loss terms.
            loss_evaluator (Callable): Python function or class which evaluates the 
                loss function.
            operator (Callable): Model being trained.
        Returns:
            List[torch.Tensor]: List of loss updated loss terms.
        """

        residual_grads = list()

        for res in residual:
            res_loss = res
            residual_grads.append(self._grad(loss=res_loss, operator=operator))

        losses_std = [torch.std(l) for l in residual_grads]

        nominator = torch.max(torch.Tensor(losses_std))

        for j in range(len(residual)):
            weight_update = self._coeff_update(
                nominator=nominator, loss=residual_grads[j]
            )

            self.weights[j] = (self.alpha) * self.weights[j] + (
                1 - self.alpha
            ) * weight_update

        return self.weights[: len(residual)]
