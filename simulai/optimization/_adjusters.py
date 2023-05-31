from typing import List, Dict, Callable
import numpy as np
import torch


class GeometricMean:

    def __init__(self):

        pass

    def __call__(self, residual=List[torch.Tensor],
                       loss_evaluator=Callable,
                       loss_history=Dict[str, float]) -> None:

        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        mean_exps = torch.mean(torch.Tensor(exps))
        shifts = [mean_exps - exp for exp in exps]

        weights = [torch.exp(shift) for shift in shifts]

        return weights

class ShiftToMax:

    def __init__(self):

        pass

    def __call__(self, residual=List[torch.Tensor],
                       loss_evaluator=Callable,
                       loss_history=Dict[str, float]) -> None:

        n_res = len(residual)
        residual_norms = [loss_evaluator(res).detach() for res in residual]
        exps = [torch.log(res) for res in residual_norms]
        max_exps = torch.max(torch.Tensor(exps))
        shifts = [max_exps - exp for exp in exps]

        weights = [torch.exp(shift)/n_res for shift in shifts]

        return weights

