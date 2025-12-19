"""
Neural network architecture for the DeepFork chess agent.

This module defines an AlphaZero-style convolutional residual network with
separate value and policy heads operating on an 8x8 board representation
constructed elsewhere in the project.
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Initial convolutional block processing the input state tensor."""

    def __init__(self, history_size=8, filter_count=256):
        super(ConvBlock, self).__init__()
        self.history_size = history_size
        self.conv = nn.Conv2d(14 * history_size + 8, filter_count, 3, padding=1)
        self.bn = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        """
        :param data: Input tensor of shape (batch, (14*h + 7)*8*8) or already (batch, channels, 8, 8)
        :return: Feature map after a conv + BN + ReLU
        """
        data = data.view(-1, 14 * self.history_size + 8, 8, 8)  # batch-size, channels, board_w, board_h
        return F.relu(self.bn(self.conv(data)))


class ResBlock(nn.Module):
    """A standard residual block with two 3x3 convolutions."""

    def __init__(self, filter_count=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.conv2 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        """
        :param data: Feature map tensor
        :return: Feature map after residual addition and ReLU
        """
        res = data
        data = F.relu(self.bn1(self.conv1(data)))
        data = self.bn2(self.conv2(data))
        data += res
        return F.relu(data)


class PolicyOutBlock(nn.Module):
    """Output heads: scalar value and flattened policy logits over 73x8x8."""

    def __init__(self, filter_count=256):
        super(PolicyOutBlock, self).__init__()
        self.convP = nn.Conv2d(filter_count, 73, 1)
        self.bnP = nn.BatchNorm2d(73)

    def forward(self, data):
        """
        :param data: Feature map tensor
        :return: policy_log_probs
        """

        p = self.bnP(self.convP(data))
        p = p.view(p.size(0), -1)

        return p


class ValueOutBlock(nn.Module):
    def __init__(self, filter_count=256):
        super(ValueOutBlock, self).__init__()
        self.convV = nn.Conv2d(filter_count, 1, 1)
        self.bnV = nn.BatchNorm2d(1)
        self.lnV1 = nn.Linear(1 * 8 * 8, 256)
        self.lnV2 = nn.Linear(256, 3)

    def forward(self, data):
        v = F.relu(self.bnV(self.convV(data)))
        v = F.relu(self.lnV1(v.view(-1, 8 * 8)))
        v = self.lnV2(v)

        return v



class DeepForkNet(nn.Module):
    """
    Residual convolutional network with AlphaZero-style heads for policy and value.

    :param depth: Number of residual blocks
    :param filter_count: Channel width for feature maps
    :param history_size: Number of historical board states encoded in input
    """

    def __init__(self, head: str, depth=5, filter_count=128, history_size=1):
        super(DeepForkNet, self).__init__()
        self.filter_count = filter_count
        self.depth = depth
        self.conv_block = ConvBlock(history_size=history_size, filter_count=filter_count)
        self.res_blocks = nn.ModuleList([ResBlock(filter_count) for _ in range(depth)])
        if head == "policy":
            self.out_block = PolicyOutBlock(filter_count=filter_count)
        else:
            self.out_block = ValueOutBlock(filter_count=filter_count)

    def forward(self, data):
        """
        :param data: Input board tensor of shape (batch, (14*h + 7)*8*8)
        :return: Tuple (value, policy_log_probs)
        """
        data = self.conv_block(data)
        for block in self.res_blocks:
            data = block(data)
        p = self.out_block(data)
        return p