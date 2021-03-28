from enum import Enum


class InitWeightsType(Enum):
    Uniformly = 1
    NormalDistribution = 2


use_batch_norm = False
use_dropout = False
dropout_keep_probability = 0.5
init_weights_type = InitWeightsType.Uniformly


def initialize() -> None:
    global use_batch_norm
    global use_dropout
    global dropout_keep_probability
    global init_weights_type
    use_batch_norm = False
    use_dropout = False
    dropout_keep_probability = 0.5
    init_weights_type = InitWeightsType.Uniformly

