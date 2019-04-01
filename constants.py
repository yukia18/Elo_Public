import random
import numpy as np
from torch import cuda


SEED = 22
DEVICE = 'gpu' if cuda.is_available() else 'cpu'

np.random.seed(SEED)
random.seed(SEED)

KEYCOL = 'card_id'
TARGETCOL = 'target'