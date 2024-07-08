import time
import numpy as np
import statistics
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table


def Flops(model, height, width, device): 
  model.eval()
  with torch.no_grad(): 
    image = torch.zeros((1, 3, height, width)).to(device)
    flops = FlopCountAnalysis(model, image)
  flops_CT = flop_count_table(flops)
  print(flops_CT)
  return flops, flops_CT

def Latency_FPS(model, height, width, device):
  image = torch.rand((1, 3, height, width)).to(device)
  iterations = 1000
  latency = []
  FPS = []
  model.eval()
  with torch.no_grad():  
    for i in range(iterations):
      start = time.time()
    
      output = model(image)
        
      end = time.time()
      ltc_i = end-start
      latency.append(ltc_i)
      FPS_i = 1/ltc_i
      FPS.append(FPS_i)
  
  meanLatency = statistics.mean(latency)*1000
  stdLatency = statistics.stdev(latency)*1000
  meanFPS = statistics.mean(FPS)
  stdFPS = statistics.stdev(FPS)
  return meanLatency, stdLatency, meanFPS, stdFPS

