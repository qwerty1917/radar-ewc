# SKKU M.IN.D RADAR EWC

***

## Installation

### Requirements
1. torch, torchvision
1. numpy
1. argparse
1. visdom
1. pillow

### Repository clone
```bash
git clone https://github.com/qwerty1917/radar-ewc.git
```

### Make virtual environment
Get into cloned directory
```bash
cd radar-ewc
```

Check current python version and get executable path.
```bash
python -c 'import sys; print(sys.executable)'
```

Make virtualenv with prechecked python executable path.
```bash
virtualenv -p [/path/prechecked/to/python] .radar-ewc-env
```

### (optional) autoenv setting
autoenv must be preinstalled.

Create `.env` file and write activation code
```bash
touch .env
echo ". .radar-ewc-env/bin/activate" > .env
```

When you exit current working folder and reenter, you will see this warning below, that you can just type `y` 
```
autoenv:
autoenv: WARNING:
autoenv: This is the first time you are about to source /mnt/user/hyeongminpark/mind/radar-ewc/.env:
autoenv:
autoenv:   --- (begin contents) ---------------------------------------
autoenv:     . .radar-ewc-env/bin/activate$
autoenv:
autoenv:   --- (end contents) -----------------------------------------
autoenv:
autoenv: Are you sure you want to allow this? (y/N) y
```

### Install requirements
```bash
pip install torch torchvision numpy argparse visdom pillow
```

## Usage

### Visdom
Create screen for visdom
```bash
screen -S visdom
```

Run visdom server
```bash
visdom -p 8085
```

Exit visdom screen
```bash
ctrl A, D
```


### Create screen for training
```bash
screen -S radar-ewc
```


### Train model
```bash
python main.py --cuda=true --multi_gpu=true --image_size=128 --env_name=baseline-001 --load_ckpt=false --reset_env=true --epoch=200 --continual=false --ewc=false --lamb=0.5 --online=false --date=20190802

```