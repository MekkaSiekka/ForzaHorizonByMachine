# Train a CNN using pytorch to play Forza Horizion 4


## Setup virtual environment

I mean if ou wanna play forza you gotta use a windows machine, right?

On main directory build a virtualenv on powershell. I call it `venv`

`./venv/Scripts/activate`

then install all dependencies using pip3.

## Install all dependencies

`pip3 install -r requirements.txt`

Additionally, install pytorch and torchvision, ref `https://pytorch.org/get-started/locally/`

CPU:

`pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`

GPU (note cuda version):

`pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`


## Usage

On main directory run

`python capture_cpu_demo.py`


