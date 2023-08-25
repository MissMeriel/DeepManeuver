#!/usr/bin/bash

python3.8 -m venv .venv-dm
. .venv-dm/bin/activate
pip install -r portalreqs.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --upgrade torch torchvision
pip install scikit-learn