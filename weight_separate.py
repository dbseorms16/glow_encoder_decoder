import os
import math
import torch
import torch.nn as nn
import sys

model_path = 'C:\\Users\\User\\Documents\\DRN\\trained_models\\face_trained_x4.pt'
weight_path = 'C:\\Users\\User\\Desktop\\LAB\\Weight'
params = torch.load(model_path)

sub_mean = {}
head = {}
down = {}
up_block_0 = {}
up_block_1 = {}
tail = {}
add_mean = {}

for k, v in params.items():
    k = str(k)
    if 'sub' in k:
        sub_mean = {k:v}
    elif 'head' in k:
        head = {k:v}
    elif 'up_block.0' in k:
        up_block_0 = {k:v}
    elif 'up_block.1' in k:
        up_block_1 = {k:v}
    elif 'tail' in k:
        sub_mean = {k:v}
    elif 'add_mean' in k:
        add_mean = {k:v}

torch.save(sub_mean, os.path.join(weight_path+'sub_mean.pt'))
torch.save(head, os.path.join(weight_path+'head.pt'))
torch.save(down, os.path.join(weight_path+'down.pt'))
torch.save(up_block_0, os.path.join(weight_path+'up_block_0.pt'))
torch.save(up_block_1, os.path.join(weight_path+'up_block_1.pt'))
torch.save(tail, os.path.join(weight_path+'tail.pt'))
torch.save(add_mean, os.path.join(weight_path+'add_mean.pt'))