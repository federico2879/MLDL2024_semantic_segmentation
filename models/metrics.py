from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
import numpy as np

def Flops(model, height, width):  
  image = torch.zeros((3, height, width))
  flops = FlopCountAnalysis(model, image)
  flops_CT = flop_count_table(flops)
  print(flops_CT)
  return flops, flops_CT

def Latency_FPS(model, height, width):
  image = torch.rand((3, height, width))
  iterations = 1000
  latency = []
  FPS = []
  
  for i in range(iterations):
    start = time.time()
    output = model(image)
    end = time.time()
    ltc_i = end-start
    latency.append(ltc_i)
    FPS_i = 1/ltc_i
    FPS.append(FPS_i)

meanLatency = mean(latency)*1000
stdLatency = mstd(latency)*1000
meanFPS = mean(FPS)*1000
stdFPS = mstd(latency)*1000
return meanLatency, stdLatency, meanFPS, stdFPS

def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def meanIOU(num_clasess, output, target):
  IOU_classes = []
  for k in range(num_classes):
    hist = []
    for i in :
      hist.append(fast_hist(pred, target, num_classes))
    
    IOU_classes.append(per_class_iou(hist))

  mIOU = sum(IOU_classes)/num_classes
  return mIOU


    
