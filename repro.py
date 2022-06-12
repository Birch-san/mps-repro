from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

netscale = 4

upsampler = RealESRGANer(
    scale=netscale,
    model_path='./Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth',
    model=realesrgan_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device='mps'
)

img = cv2.imread('./in.jpg')

enhanced, _ = upsampler.enhance(img, outscale=2)
cv2.imwrite('./out.jpg', enhanced)