from ocf_iam4vp import IAM4VP as IAM4VPBase
from torch import nn, stack, Tensor


class IAM4VP(nn.Module):

    def __init__(
        self, num_channels, history_len, forecast_len, hid_S=16, hid_T=256, N_S=4, N_T=8
    ):
        super().__init__()
        self.model = IAM4VPBase(
            num_channels,
            num_history_steps=history_len,
            num_forecast_steps=forecast_len,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
        )

    def forward(self, X):
        # Input batches have shape: (batch, channel, time, height, width)
        y_hats: list[Tensor] = []
        for _ in range(self.model.num_forecast_steps):
            y_hats.append(self.model(X, y_hats))
        return stack(y_hats, dim=2)
