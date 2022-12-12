


import os
import time
import string
import argparse
import re
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model, JitModel
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

opt = get_args(is_train=False)
opt.benchmark_all_eval = True
opt.Transformer = True
opt.sensitive = True
opt.data_filtering_off = True
opt.imgH = 224
opt.imgW = 224
opt.TransformerModel = "vitstr_small_patch16_224"
opt.saved_model = 'vitstr_small_patch16_jit.pt'
print(opt.Transformer)
