"""Adapted from https://github.com/A4Bio/SimVP


"""

from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class Earthformer(CuboidTransformerModel):
    
    def forward(self, X, verbose=False):
        X = X.permute(0, 2, 3, 4, 1).contiguous()
        out = super().forward(X, verbose=verbose)
        return out.permute(0, 4, 1, 2, 3)