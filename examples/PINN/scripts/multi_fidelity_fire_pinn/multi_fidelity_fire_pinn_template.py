def model_():
    from typing import List

    import torch

    torch.set_default_dtype(torch.float64)

    import numpy as np

    from simulai.models import ImprovedDenseNetwork
    from simulai.regression import SLFNN, ConvexDenseNetwork
    from simulai.templates import NetworkTemplate, guarantee_device

    delta_t = 4
    n_intervals = 50

    def sub_model():
        # Configuration for the fully-connected network
        config = {
            "layers_units": [50, 50, 50],
            "activations": "tanh",
            "input_size": 1,
            "output_size": 1,
            "name": "net",
        }

        # Instantiating and training the surrogate model
        densenet = ConvexDenseNetwork(**config)
        encoder_u = SLFNN(input_size=1, output_size=50, activation="tanh")
        encoder_v = SLFNN(input_size=1, output_size=50, activation="tanh")

        net = ImprovedDenseNetwork(
            network=densenet,
            encoder_u=encoder_u,
            encoder_v=encoder_v,
            devices="gpu",
        )

        return net

    models_list = [sub_model() for i in range(n_intervals)]

    class MultiNetwork(NetworkTemplate):
        def __init__(
            self,
            models_list: List[NetworkTemplate] = None,
            delta_t: float = None,
            device: str = "cpu",
        ) -> None:
            super(MultiNetwork, self).__init__()

            for i, model in enumerate(models_list):
                self.set_network(net=model, index=i)

            self.delta_t = delta_t
            self.device = device

        def set_network(self, net: NetworkTemplate = None, index: int = None) -> None:
            setattr(self, f"worker_{index}", net)

            self.add_module(f"worker_{index}", net)

        def _eval_interval(
            self, index: int = None, input_data: torch.Tensor = None
        ) -> torch.Tensor:
            input_data = np.array([input_data])[:, None]
            return getattr(self, f"worker_{index}").eval(input_data=input_data)

        def eval(self, input_data: np.ndarray = None) -> np.ndarray:
            eval_indices_float = input_data / delta_t
            eval_indices = np.where(
                eval_indices_float > 0,
                np.floor(eval_indices_float - 1e-13).astype(int),
                eval_indices_float.astype(int),
            )

            input_data = input_data - self.delta_t * eval_indices

            eval_interval = np.vectorize(self._eval_interval)

            return eval_interval(index=eval_indices, input_data=input_data)

        def summary(self):
            print(self)

    multi_net = MultiNetwork(models_list=models_list, delta_t=delta_t)

    return multi_net
