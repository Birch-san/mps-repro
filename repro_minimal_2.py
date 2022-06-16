import numpy as np
from numpy import float32, array
import torch as th
from torch import tensor, allclose

arr = array([[[0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        ]],

       [[0.        , 0.        , 0.        ],
        [0.49803922, 0.49803922, 0.49803922],
        [1.        , 1.        , 1.        ]],

       [[0.        , 0.        , 0.        ],
        [1.        , 1.        , 1.        ],
        [1.        , 1.        , 1.        ]]], dtype=float32)

def get_tensor():
  arr_t = np.transpose(arr, (2, 0, 1))
  img = th.from_numpy(arr_t).float()
  img = img.unsqueeze(0)
  return img

hardcoded_tensor = tensor([[[[0.0000, 0.0000, 0.0000],
  [0.0000, 0.4980, 1.0000],
  [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
  [0.0000, 0.4980, 1.0000],
  [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
  [0.0000, 0.4980, 1.0000],
  [0.0000, 1.0000, 1.0000]]]])

unsqueezed_via_cpu = hardcoded_tensor.to('cpu')
unsqueezed_via_mps = hardcoded_tensor.to('mps').cpu()
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "expect {hardcoded_tensor -> CPU} == {hardcoded_tensor -> GPU -> CPU}"

computed_tensor = get_tensor()
print('hardcoded_tensor:\n', hardcoded_tensor)
print('computed_tensor:\n', computed_tensor)
assert allclose(computed_tensor, hardcoded_tensor, rtol=0.0001), "expect hardcoded_tensor == get_tensor()"

unsqueezed_via_cpu = computed_tensor.to('cpu')
unsqueezed_via_mps = computed_tensor.to('mps').cpu()
print('unsqueezed_via_cpu:\n', unsqueezed_via_cpu)
print('unsqueezed_via_mps:\n', unsqueezed_via_mps)
assert allclose(unsqueezed_via_cpu, unsqueezed_via_mps, rtol=0.0001), "expect {get_tensor() -> CPU} == {get_tensor() -> GPU -> CPU}" # fails!