from numpy import array_equal
from torch import tensor
import torch as th

input = tensor(
  [[[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]],

  [[0.0000, 0.0000, 0.0000],
    [0.0000, 0.4980, 1.0000],
    [0.0000, 1.0000, 1.0000]]])

cpu_output = input.unsqueeze(0).to('cpu')
mps_output = input.unsqueeze(0).to('mps')

th.save(cpu_output, './out_cpu_repro.pt')
th.save(mps_output, './out_mps_repro.pt')

cpu_roundtrip = th.load('./out_cpu_repro.pt')
mps_roundtrip = th.load('./out_mps_repro.pt')
# assert array_equal(cpu_output.numpy(), mps_output.cpu().numpy()), "unsqueezed MPS tensor differs from CPU counterpart"
assert array_equal(cpu_roundtrip, mps_roundtrip), "unsqueezed MPS tensor differs from CPU counterpart"