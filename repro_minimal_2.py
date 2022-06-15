import numpy as np
from numpy import array_equal, vectorize, uint8
import torch as th
from torch import tensor
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

def choose_colour (row_ix: uint8, col_ix: uint8, _channel_ix: uint8) -> uint8:
  if row_ix == 1 and col_ix == 1:
    return 127
  elif (row_ix == 2 and col_ix > 0) or (col_ix == 2 and row_ix > 0):
    return 255
  return 0

def make_img(device_type, half):
  scale = 4
  tile_size = 0
  tile_pad = 10
  pre_pad = 0
  mod_scale = None
  half = False
  model_path = './Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth'
  loadnet = th.load(model_path, map_location=th.device('cpu'))
  realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
  realesrgan_model.load_state_dict(loadnet['params_ema'], strict=True)
  realesrgan_model.eval()
  realesrgan_model = realesrgan_model.to(device_type)
  if half:
    realesrgan_model = realesrgan_model.half()
  img = np.fromfunction(function=vectorize(choose_colour), shape=(3, 3, 3), dtype=uint8)
  h_input, w_input = img.shape[0:2]
  img = img.astype(np.float32)
  img = img / 255
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = th.from_numpy(np.transpose(img, (2, 0, 1))).float()
  img = img.unsqueeze(0).to(device_type)
  return img

half = False
cpu_output = make_img('cpu', half)
mps_output = make_img('mps', half)
assert array_equal(cpu_output, mps_output), "unsqueezed MPS tensor differs from CPU counterpart"

# cpu_output = input.unsqueeze(0).to('cpu')
# mps_output = input.unsqueeze(0).to('mps')

# th.save(cpu_output, './out_cpu_repro.pt')
# th.save(mps_output, './out_mps_repro.pt')

# cpu_roundtrip = th.load('./out_cpu_repro.pt')
# mps_roundtrip = th.load('./out_mps_repro.pt')
# assert array_equal(cpu_output.numpy(), mps_output.cpu().numpy()), "unsqueezed MPS tensor differs from CPU counterpart"
# assert array_equal(cpu_roundtrip, mps_roundtrip), "unsqueezed MPS tensor differs from CPU counterpart"