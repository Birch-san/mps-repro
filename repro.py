import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

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

img = cv2.imread('./in.jpg')

enhanced, _ = upsampler.enhance(img, outscale=2)
cv2.imwrite(f'./out_{args.backend_type}_half_{str(args.half_precision_float)}.jpg', enhanced)