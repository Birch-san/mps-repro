```bash
git submodule update --init --recursive
git submodule add https://github.com/xinntao/Real-ESRGAN.git
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P Real-ESRGAN/experiments/pretrained_models
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip
pip install basicsr facexlib git+https://github.com/Birch-san/GFPGAN.git@newer-numpy
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
cd ..
# our torch nightly probably got nuked by the above, but we do need it for GPU support on macOS
pip install --pre "torch==1.13.0.dev20220610" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python repro.py
```