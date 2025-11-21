import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block used for processing input data in a neural network.

    The ConvBlock class is a building block for convolutional neural networks.
    It applies a 2D convolution followed by batch normalization and a ReLU activation
    to transform the input data. The class is designed to handle input with specific
    dimensions and structure and is particularly useful in tasks requiring a
    history-sensitive model.

    :ivar history_size: Number of historical frames combined with the base
        input data across channels.
    :type history_size: int
    :ivar conv: 2D convolutional layer that operates on the transformed data
        and applies filters.
    :type conv: nn.Conv2d
    :ivar bn: Batch normalization layer applied after convolution.
    :type bn: nn.BatchNorm2d
    """
    def __init__(self, history_size=8, filter_count=256):
        super(ConvBlock, self).__init__()
        self.history_size = history_size
        self.conv = nn.Conv2d(14 * history_size + 7, filter_count, 3, padding=1)
        self.bn = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        data = data.view(-1, 14 * self.history_size + 7, 8, 8) # batch-size, channels, board_w, board_h
        return F.relu(self.bn(self.conv(data)))


class ResBlock(nn.Module):
    """
    Residual Block for a neural network.

    This class defines a residual block. It performs two stages of convolution followed by batch
    normalization and applies the ReLU activation function. The residual
    skip connection is added to facilitate gradient flow and improve
    training in deeper networks.

    :ivar conv1: First convolutional layer with kernel size 3x3 and padding 1.
    :type conv1: nn.Conv2d
    :ivar bn1: Batch normalization layer corresponding to conv1.
    :type bn1: nn.BatchNorm2d
    :ivar conv2: Second convolutional layer with kernel size 3x3 and padding 1.
    :type conv2: nn.Conv2d
    :ivar bn2: Batch normalization layer corresponding to conv2.
    :type bn2: nn.BatchNorm2d
    """
    def __init__(self, filter_count=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_count)
        self.conv2 = nn.Conv2d(filter_count, filter_count, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_count)

    def forward(self, data):
        res = data
        data = F.relu(self.bn1(self.conv1(data)))
        data = F.relu(self.bn2(self.conv2(data)))
        data += res
        return F.relu(data)


class OutBlock(nn.Module):
    """
    Defines the OutBlock class, which represents an output block in a neural
    network.

    This block is designed to output both a value prediction (scalar) and a
    policy prediction (probability distribution) from the network. It consists
    of separate heads for value and policy computation, each with its own set
    of convolutional, batch normalization, linear, and activation layers.

    :ivar convV: Convolutional layer for the value head.
    :type convV: torch.nn.Conv2d
    :ivar bnV: Batch normalization layer for the value head.
    :type bnV: torch.nn.BatchNorm2d
    :ivar lnV1: First linear layer in the value head.
    :type lnV1: torch.nn.Linear
    :ivar lnV2: Second linear layer in the value head that outputs a single value.
    :type lnV2: torch.nn.Linear
    :ivar convP: Convolutional layer for the policy head.
    :type convP: torch.nn.Conv2d
    :ivar bnP: Batch normalization layer for the policy head.
    :type bnP: torch.nn.BatchNorm2d
    :ivar lsmP: Log-softmax activation for the policy head.
    :type lsmP: torch.nn.LogSoftmax
    """
    def __init__(self, filter_count=256):
        super(OutBlock, self).__init__()
        # Value head
        self.convV = nn.Conv2d(filter_count, 1, 1)
        self.bnV = nn.BatchNorm2d(1)
        self.lnV1 = nn.Linear(8 * 8, 256)
        self.lnV2 = nn.Linear(256, 1)

        # Policy head
        self.convP = nn.Conv2d(filter_count, 73, 1)
        self.bnP = nn.BatchNorm2d(73)
        self.lsmP = nn.LogSoftmax(dim=1)

    def forward(self, data):
        # Value head
        v = F.relu(self.bnV(self.convV(data)))
        v = F.relu(self.lnV1(v.view(-1, 8 * 8)))
        v = F.tanh(self.lnV2(v))

        # Policy head
        p = F.relu(self.bnP(self.convP(data)))
        p = self.lsmP(p.view(-1, 73 * 8 * 8))

        return v, p


class DeepForkNet(nn.Module):
    """
    DeepForkNet is a neural network model designed for specific deep learning tasks.

    This model is structured with an initial convolutional block, followed by multiple
    residual blocks and an output block. It is suitable for processing inputs with a given
    history size and performing tasks requiring both high-level features and detailed
    prediction outputs.

    :ivar filter_count: Number of filters used in each layer of the network.
    :type filter_count: int
    :ivar depth: Number of residual blocks in the network.
    :type depth: int
    :ivar conv_block: Initial convolutional block used to process the input data.
    :type conv_block: ConvBlock
    :ivar res_blocks: List of residual blocks for feature extraction.
    :type res_blocks: nn.ModuleList
    :ivar out_block: Final output block generating the predictions.
    :type out_block: OutBlock
    """
    def __init__(self, depth=10, filter_count=256, history_size=8):
        super(DeepForkNet, self).__init__()
        self.filter_count = filter_count
        self.depth = depth
        self.conv_block = ConvBlock(history_size=history_size, filter_count=filter_count)
        self.res_blocks = nn.ModuleList([ResBlock(filter_count) for _ in range(depth)])
        self.out_block = OutBlock(filter_count=filter_count)

    def forward(self, data):
        data = self.conv_block(data)
        for block in self.res_blocks:
            data = block(data)
        v, p = self.out_block(data)
        return v, p