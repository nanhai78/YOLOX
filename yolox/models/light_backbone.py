from torch import nn
from .network_blocks import BaseConv, RepVGGBlock, ES_Block1, SPPF
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, \
    SPPBottleneck

"""
    stem => 6 * 6卷积
    act => hard_swish
    dark => (rep_conv, es_block)
    spp =>sppf
"""


class PicoNet(nn.Module):
    def __init__(
            self,
            wid_mul,
            base_depth=None,
            out_features=("dark3", "dark4", "dark5"),
            act="hard_swish",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        if base_depth is None:
            base_depth = [1, 3, 7, 3]  # repeat number of es_block
        self.out_features = out_features
        base_channels = int(wid_mul * 64)  # 64

        # stem
        self.stem = BaseConv(3, base_channels, ksize=6, stride=2, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            # BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            ES_Block2(base_channels, base_channels * 2),
            *[ES_Block1(base_channels * 2, base_channels * 2) for _ in range(base_depth[0])]
        )

        # dark3
        self.dark3 = nn.Sequential(
            # BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            ES_Block2(base_channels * 2, base_channels * 4),
            *[ES_Block1(base_channels * 4, base_channels * 4) for _ in range(base_depth[1])]
        )

        # dark4
        self.dark4 = nn.Sequential(
            # BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            ES_Block2(base_channels * 4, base_channels * 8),
            *[ES_Block1(base_channels * 8, base_channels * 8) for _ in range(base_depth[2])]
        )
        self.dark5 = nn.Sequential(
            # BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            ES_Block2(base_channels * 8, base_channels * 16),
            *[ES_Block1(base_channels * 16, base_channels * 16) for _ in range(base_depth[3])],  # csp
            SPPF(base_channels * 16, base_channels * 16, activation=act)  # spp
        )
        # dark5

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


class NewDarknet(nn.Module):
    def __init__(
            self,
            dep_mul,
            wid_mul,
            out_features=("dark3", "dark4", "dark5"),
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            ES_Block1(
                base_channels * 4,
                base_channels * 4,
                k_sizes=[3, 5, 7],
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            ES_Block1(
                base_channels * 8,
                base_channels * 8,
                k_sizes=[3, 7, 11],
                SE=True,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            ES_Block1
                (
                base_channels * 16,
                base_channels * 16,
                k_sizes=[5, 9, 13],
                SE=True,
                act=act,
            ),
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
