import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
# from cv2 import Mat, CV_8UC3
import numpy as np
from numpy import uint8, vectorize # fromfunction ndenumerate

realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0', ''):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--half_precision_float', type=str2bool, default=False, choices=[True, False])
parser.add_argument('--backend_type', type=str, default='cpu', choices=['cpu', 'mps'])
args = parser.parse_args()

upsampler = RealESRGANer(
    scale=4,
    model_path='./Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth',
    model=realesrgan_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=args.half_precision_float,
    device=args.backend_type
)

def choose_colour (row_ix: uint8, col_ix: uint8, channel_ix: uint8) -> uint8:
  if row_ix == 1 and col_ix == 1:
    return 127
  elif (row_ix == 2 and col_ix > 0) or (col_ix == 2 and row_ix > 0):
    return 255
  return 0

img = np.fromfunction(function=vectorize(choose_colour), shape=(3, 3, 3), dtype=uint8)

# for (row_ix, col_ix, _channel_ix), cell in ndenumerate(img):
#     if row_ix == 1 and col_ix == 1:
#       cell[0] = 127
#       cell[1] = 127
#       cell[2] = 127
#     elif (row_ix == 2 and col_ix > 1) or (col_ix == 2 and row_ix > 1):
#       cell[0] = 255
#       cell[1] = 255
#       cell[2] = 255
cv2.imwrite(f'./out_dbg_{args.backend_type}_half_{str(args.half_precision_float)}.pre.jpg', img)

# img = cv2.imread('./in.jpg')

# def gray(): 
# array([127, 127, 127], dtype=uint8)

enhanced, _ = upsampler.enhance(img, outscale=2)
cv2.imwrite(f'./out_dbg_{args.backend_type}_half_{str(args.half_precision_float)}.jpg', enhanced)