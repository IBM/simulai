from typing import Dict, Union
import numpy as np
import torch


def view_api(
    module: torch.nn.Module = None,
    input_data: Union[torch.nn.Module, np.ndarray] = None,
    config: Dict = None,
    save_dir: str = None,
):
    try:
        from torchview import draw_graph
    except Exception:
        raise Exception(
            "For using the visualization API,\
                        it is necessary to install the torchview module."
        )

    # Wrapper for torchview
    class Module_wrap(torch.nn.Module):
        def __init__(self, module=None):
            super(Module_wrap, self).__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args)

        def train(self, *args, **kwargs):
            pass

        def eval(self, *args, **kwargs):
            pass

    module_wrap = Module_wrap(module=module)

    model_graph = draw_graph(
        module_wrap, input_data=input_data, directory=save_dir, **config
    )

    return model_graph.visual_graph
