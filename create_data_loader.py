import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import glob
import pandas as pd
import json
import os

SEQ_LEN = 64
STEP_SIZE = 8


def create_y(l, h, o):
    if o * 1.01 < h and o * 0.99 > l:
        return 0
    elif o * 1.01 > h:
        return 1
    else:
        return 2


def high_low(high, low):
    return lambda x: (x - low) / (high - low)


# https://diegslva.github.io/2017-05-02-first-post/
def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0, window_size, step_size)


path = 'data/DAY/'
rollings = []
for fname in os.listdir(path):
    print(fname)
    fname = path + fname
    with open(fname) as f:
        data = json.load(f)
        if not data:
            continue
        dataFrame = pd.DataFrame(data).dropna()
        d_high = torch.Tensor(dataFrame['high'])
        d_low = torch.Tensor(dataFrame['low'])
        normalize = high_low(torch.max(d_high), torch.min(d_low))
        d_high = normalize(d_high)
        d_low = normalize(d_low)
        d_y = [create_y(l, h, o) for l, h, o in zip(d_low, d_high, d_open)]
        d_y = torch.Tensor([d_y[-1]] + d_y[:-1])
        d_open = normalize(torch.Tensor(dataFrame['open']))
        d_close = normalize(torch.Tensor(dataFrame['close']))
        d_volume = torch.Tensor(dataFrame['volume'])
        d_time = torch.Tensor(dataFrame['timestamp'])
        matrix = torch.stack([d_time, d_low, d_high, d_open, d_close,
                              d_volume, d_y]).transpose(0, 1)
        rolled = pytorch_rolling_window(matrix, SEQ_LEN, STEP_SIZE)
        rollings.append(rolled)


# make a range sequence sample
x = torch.range(1, 20)
# ie. window size of 5, step size of 1
print(pytorch_rolling_window(x, 5, 1))

torch.utils.data.dataloader
