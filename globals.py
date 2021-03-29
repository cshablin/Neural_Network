from enum import Enum


class InitWeightsType(Enum):
    Uniformly = 1
    NormalDistribution = 2


use_batch_norm = False
use_dropout = False
dropout_keep_probability = 0.8
MAXIMUM_EPOCH_COUNT = 1000
init_weights_type = InitWeightsType.NormalDistribution
x_val = None
y_val = None


