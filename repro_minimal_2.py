import numpy as np
from numpy import vectorize, uint8
import torch as th
from torch import tensor, allclose
import cv2

def choose_colour (row_ix: uint8, col_ix: uint8, _channel_ix: uint8) -> uint8:
  if row_ix == 1 and col_ix == 1:
    return 127
  elif (row_ix == 2 and col_ix > 0) or (col_ix == 2 and row_ix > 0):
    return 255
  return 0

def get_tensor():
  img = np.fromfunction(function=vectorize(choose_colour), shape=(3, 3, 3), dtype=uint8)
  img = img.astype(np.float32)
  img = img / 255
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = th.from_numpy(np.transpose(img, (2, 0, 1))).float()
  return img

def unsqueeze(tensor, device_type):
  return tensor.unsqueeze(0).to(device_type)

hardcoded_tensor = tensor(
  [[[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]]])

unsqueezed_via_cpu = hardcoded_tensor.unsqueeze(0).to('cpu')
unsqueezed_via_mps = hardcoded_tensor.unsqueeze(0).to('mps').cpu()
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "unsqueezing the harcoded tensor gives the same result o both CPU and on MPS"

computed_tensor = get_tensor()
print('hardcoded_tensor:\n', hardcoded_tensor)
print('computed_tensor:\n', computed_tensor)
assert allclose(computed_tensor, hardcoded_tensor, rtol=0.0001), "the hardcoded tensor is equivalent to the one we compute via get_tensor, so we should get the same result when we unsqueeze it… right?"

unsqueezed_via_cpu = computed_tensor.unsqueeze(0).to('cpu')
unsqueezed_via_mps = computed_tensor.unsqueeze(0).to('mps').cpu()
print('unsqueezed_via_cpu:\n', unsqueezed_via_cpu)
print('unsqueezed_via_mps:\n', unsqueezed_via_mps)
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "unsqueezed MPS tensor differs from CPU counterpart"