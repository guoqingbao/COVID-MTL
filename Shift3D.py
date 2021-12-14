import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *

class Shift3DLayer(nn.Module):
    def __init__(self, shift_chance=0.25, batch_shift=False, max_shift_percentage = 0.2, decay_iterations=0, is_padding = False, paddding_value = 0):
        super(Shift3DLayer, self).__init__()
        self.shift_chance = shift_chance 
        self.max_shift_percentage = max_shift_percentage
        self.padding = is_padding
        self.paddding_value = paddding_value
        self.cur_iteration = torch.tensor(0)
        self.decay_iterations = decay_iterations
        self.batch_shift = batch_shift
        
    def shift3d(self, x):
        dim = random.randint(0,2) #dimension of the shift
        shifts = random.randint(0,int(self.max_shift_percentage * x.shape[dim])) # the number of lines to shift
        forward = -1 if random.randint(0,1) > 0 else 1 # forward or backward shift
        shifted = torch.roll(x, shifts=forward*shifts, dims=dim) # shift according to random dim, number of lines and forward/backward

        if self.padding == True: #padding with zeros
            if forward == -1: # padding for forward shift 
                if dim ==0:
                    shifted[forward*shifts:,:,:] = self.paddding_value
                elif dim==1:
                    shifted[:,forward*shifts:,:] = self.paddding_value
                elif dim==2:
                    shifted[:,:,forward*shifts:] = self.paddding_value
            else: # paddinng for backward shift 
                if dim ==0:
                    shifted[:shifts,:,:] = self.paddding_value
                elif dim==1:
                    shifted[:,:shifts,:] = self.paddding_value
                elif dim==2:
                    shifted[:,:,:shifts] = self.paddding_value
        return shifted

    def get_chance(self):
        if self.shift_chance <= 0: return 0
        if self.decay_iterations<=0: return self.shift_chance
        if self.cur_iteration > self.decay_iterations: return 0

        chance =  0.5 * self.shift_chance * (math.cos(math.pi * self.cur_iteration / self.decay_iterations) + 1)
        self.cur_iteration += 1
        return chance

    def forward(self, x):
        chance = self.get_chance()
        if chance <= 0:
            return x

        delta = int(1.0/chance)
        if not self.batch_shift:   
            for sample in range(x.shape[0]):
                if delta > 1 and random.randint(1, delta) == 1:
                    for channel in range(x.shape[1]):
                            x[sample, channel] = self.shift3d(x[sample, channel])
        else:
            if delta > 1 and random.randint(1, delta) == 1:
                for sample in range(x.shape[0]):
                    for channel in range(x.shape[1]):
                            x[sample, channel] = self.shift3d(x[sample, channel])
        return x