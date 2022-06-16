import numpy as np
from numpy import float32, array
import torch as th
from torch import tensor

arr = array([[[0, 0],
        [0, 1]],

       [[0, 0],
        [0, 1]]], dtype=float32)

def get_tensor():
  arr_t = np.transpose(arr, (2, 0, 1))
  img = th.from_numpy(arr_t)
  img = img.unsqueeze(0)
  return img

hardcoded_tensor = tensor([[[[0., 0.],
          [0., 0.]],

         [[0., 1.],
          [0., 1.]]]])

via_mps = hardcoded_tensor.to('mps').cpu()
assert th.equal(hardcoded_tensor, via_mps), "hardcoded_tensor == {hardcoded_tensor -> GPU -> CPU}"

computed_tensor = get_tensor()
print('hardcoded_tensor:\n', hardcoded_tensor)
print('computed_tensor:\n', computed_tensor)
assert th.equal(computed_tensor, hardcoded_tensor), "hardcoded_tensor == get_tensor()"

via_mps = computed_tensor.to('mps').cpu()
print('computed_on_cpu:\n', computed_tensor)
print('computed_via_mps:\n', via_mps)
assert th.equal(computed_tensor, via_mps), "get_tensor() == {get_tensor() -> GPU -> CPU}" # fails!