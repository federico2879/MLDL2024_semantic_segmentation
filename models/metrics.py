#!pip install -U fvcore

from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import numpy as np
import statistics

def metric_pip_install():
  !pip install -U fvcore


def Flops(model, height, width): 
  model.eval()
  with torch.no_grad(): 
    image = torch.zeros((1, 3, height, width)).to(device)
    flops = FlopCountAnalysis(model, image)
  flops_CT = flop_count_table(flops)
  print(flops_CT)
  return flops, flops_CT

def Latency_FPS(model, height, width):
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
  meanFPS = statistics.mean(FPS)*1000
  stdFPS = statistics.stdev(FPS)*1000
  return meanLatency, stdLatency, meanFPS, stdFPS

def fast_hist(pred, target, num_classes):
    k = (pred >= 0) & (pred < num_classes)
    return np.bincount(num_classes * pred[k].astype(int) + target[k], minlength = num_classes**2).reshape(num_classes, num_classes)

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def meanIOU(num_clasess, pred, target):
  mIOU = 0
  for i in range(len(pred)):    
      hist = fast_hist(pred[i].cpu().numpy(), target[i].cpu().numpy(), num_classes)
      IOU = per_class_iou(hist)
      mIOU = mIOU + sum(IOU)/num_classes 
  return mIOU 
