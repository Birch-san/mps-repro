import numpy as np
from numpy import array_equal, vectorize, uint8
import torch as th
from torch import tensor
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

def make_img(tensor, device_type):
  return tensor.unsqueeze(0).to(device_type)

canned_tensor = tensor(
  [[[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]]])

half = False
# input_tensor = get_tensor()
cpu_output = make_img(canned_tensor, 'cpu')
mps_output = make_img(canned_tensor, 'mps')
# cpu_output = make_img_unit('cpu')
# mps_output = make_img_unit('mps').cpu()
print('cpu_output:\n', cpu_output)
print('mps_output:\n', mps_output)
assert array_equal(cpu_output, mps_output), "unsqueezed MPS tensor differs from CPU counterpart"

# cpu_output = input.unsqueeze(0).to('cpu')
# mps_output = input.unsqueeze(0).to('mps')

# th.save(cpu_output, './out_cpu_repro.pt')
# th.save(mps_output, './out_mps_repro.pt')

# cpu_roundtrip = th.load('./out_cpu_repro.pt')
# mps_roundtrip = th.load('./out_mps_repro.pt')
# assert array_equal(cpu_output.numpy(), mps_output.cpu().numpy()), "unsqueezed MPS tensor differs from CPU counterpart"
# assert array_equal(cpu_roundtrip, mps_roundtrip), "unsqueezed MPS tensor differs from CPU counterpart"