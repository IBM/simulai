def model_():
    from typing import List

    import torch

    torch.set_default_dtype(torch.float64)

    import numpy as np

    from simulai.models import ImprovedDenseNetwork
    from simulai.regression import SLFNN, ConvexDenseNetwork
    from simulai.templates import NetworkTemplate, guarantee_device

    depth = 3
    width = 50
    activations_funct = "tanh"
    n_intervals = 150  # overestimated

    # Model used for initialization
    def sub_model():
        from simulai.models import ImprovedDenseNetwork
        from simulai.regression import SLFNN, ConvexDenseNetwork

        scale_factors = np.array([1, 1, 1, 1])

        # Configuration for the fully-connected network
        config = {
            "layers_units": depth * [width],  # Hidden layers
            "activations": activations_funct,
            "input_size": 1,
            "output_size": 4,
            "name": "net",
        }

        # Instantiating and training the surrogate model
        densenet = ConvexDenseNetwork(**config)
        encoder_u = SLFNN(input_size=1, output_size=width, activation=activations_funct)
        encoder_v = SLFNN(input_size=1, output_size=width, activation=activations_funct)

        class ScaledImprovedDenseNetwork(ImprovedDenseNetwork):
            def __init__(
                self,
                network=None,
                encoder_u=None,
                encoder_v=None,
                devices="gpu",
                scale_factors=None,
            ):
                super(ScaledImprovedDenseNetwork, self).__init__(
                    network=densenet,
                    encoder_u=encoder_u,
                    encoder_v=encoder_v,
                    devices="gpu",
                )
                self.scale_factors = torch.from_numpy(
                    scale_factors.astype("float32")
                ).to(self.device)

            def forward(self, input_data=None):
                return super().forward(input_data) * self.scale_factors

        net = ScaledImprovedDenseNetwork(
            network=densenet,
            encoder_u=encoder_u,
            encoder_v=encoder_v,
            devices="gpu",
            scale_factors=scale_factors,
        )

        # It prints a summary of the network features
        # net.summary()

        return net

    # Initialization for the multifidelity network
    models_list = [sub_model() for i in range(n_intervals)]

    # Prototype for the multifidelity network
    class MultiNetwork(NetworkTemplate):
        def __init__(
            self, models_list: List[NetworkTemplate] = None, device: str = "cpu"
        ) -> None:
            super(MultiNetwork, self).__init__()

            for i, model in enumerate(models_list):
                self.set_network(net=model, index=i)

            self.delta_t = torch.nn.Parameter(
                data=torch.zeros(len(models_list)), requires_grad=False
            ).to(device)

            self.device = device

        def set_delta_t(self, delta_t: float, index: int = None) -> None:
            getattr(self, "delta_t").data[index] = delta_t

        def set_network(self, net: NetworkTemplate = None, index: int = None) -> None:
            setattr(self, f"worker_{index}", net)

            self.add_module(f"worker_{index}", net)

        def _eval_interval(
            self, index: int = None, input_data: torch.Tensor = None
        ) -> torch.Tensor:
            input_data = input_data[:, None]
            return getattr(self, f"worker_{index}").eval(input_data=input_data)

        @property
        def time_intervals(self):
            cumulated_time = [0] + np.cumsum(self.delta_t).tolist()

            return np.array(
                [
                    [cumulated_time[i], cumulated_time[i + 1]]
                    for i in range(len(cumulated_time) - 1)
                ]
            )

        def _interval_delimiter(self, value: float, lower: float, upper: float) -> bool:
            if lower <= value[0] < upper:
                return True
            else:
                return False

        def eval(self, input_data: np.ndarray = None) -> np.ndarray:
            n_samples = input_data.shape[0]

            time_intervals = self.time_intervals

            eval_indices = list()
            for j in range(n_samples):
                is_in_interval = [
                    self._interval_delimiter(input_data[j], low, up)
                    for low, up in time_intervals.tolist()
                ]
                index = is_in_interval.index(True)
                eval_indices.append(index)

            cumulated_time = np.array([0] + np.cumsum(self.delta_t).tolist())

            input_data = input_data - cumulated_time[np.array(eval_indices)][:, None]

            return np.vstack(
                [
                    self._eval_interval(index=i, input_data=idata)
                    for i, idata in zip(eval_indices, input_data)
                ]
            )

        def summary(self):
            print(self)

    multi_net = MultiNetwork(models_list=models_list)

    return multi_net
