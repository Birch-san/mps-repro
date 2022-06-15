import numpy as np
import torch as th

half = False

for ix in range(1, 3):
  cpu = th.load(f'./out/dbg_cpu_half_{half}/pre{ix}.pt')
  mps = th.load(f'./out/dbg_mps_half_{half}/pre{ix}.pt')
  assert np.array_equal(cpu, mps), ix

for ix in range(1, 11):
  cpu = th.load(f'./out/dbg_cpu_half_{half}/enh{ix}.pt')
  mps = th.load(f'./out/dbg_mps_half_{half}/enh{ix}.pt')
  assert np.array_equal(cpu, mps), ix