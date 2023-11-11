from torch import nn
from .network_blocks import BaseConv, RepVGGBlock, ES_Block

"""
    stem => 6 * 6卷积
    act => hard_swish
    dark => (rep_conv, es_block)
"""
class CSPDarknet(nn.Module):
    def __init__(
            self,
            dep_mul,
            wid_mul,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="hard_swish",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3 基本深度

        # stem
        self.stem = BaseConv(3, base_channels, ksize=6, stride=2, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            RepVGGBlock(base_channels, base_channels * 2, 3, 2, act=act),
            ES_Block(base_channels * 2, base_channels * 2, 1)
        )

        # dark3
        self.dark3 = nn.Sequential(
            RepVGGBlock(base_channels * 2, base_channels * 4, 3, 2, act=act),
            ES_Block(base_channels * 4, base_channels * 4, 1)
        )

        # dark4
        self.dark4 = nn.Sequential(
            RepVGGBlock(base_channels * 4, base_channels * 8, 3, 2, act=act),
            ES_Block(base_channels * 8, base_channels * 8, 1)
        )

        # dark5
        self.dark5 = nn.Sequential(
            RepVGGBlock(base_channels * 8, base_channels * 16, 3, 2, act=act),
            # spp
            # csp
            ES_Block(base_channels * 16, base_channels * 16, 1)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}