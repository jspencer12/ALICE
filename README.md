# ALICE
## Aggregating Losses to Imitate Cached Experts

This code serves as a general purpose framework for implementing the ALICE imitation learning algorithm as well as several imitation learning baselines. Our training of "Experts" uses the training code from [rl-baselines-zoo](https://github.com/araffin/rl-baselines-zoo), but it's super flexible, and with some modification to the expert loading method should work with most "expert" models. Cached trajectories and results are stored via pandas for flexible pruning/sorting/plotting.

## Installation Instructions:

Requires python 3.6, and for baselines you should stick with tensorflow 1.14 or 1.15

1. Install virtual env (Python 3.6 on Ubuntu)
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3.6-venv
```
2. Make the virtual environment (rename "venv_ALICE" with whatever you want)
```
python3.6 -m venv venv_ALICE
```
3. Activate the venv (Type 'deactivate' to deactivate session)
```
. venv_ALICE/bin/activate
```
4. Install dependencies
Perhaps need:
```
apt-get install swig cmake libopenmpi-dev zlib1g-dev ffmpeg
```
Definitely need
```
pip install --upgrade pip
pip install -r requirements.txt
```
5. Install mujoco (if you're into that.) Be sure to place mjkey.txt into the ~/.mujoco/ folder.
```
wget https://www.roboti.us/download/mjpro150_linux.zip
mkdir ~/.mujoco
unzip mjpro150_linux.zip -d ~/.mujoco
sudo apt install python3.x-dev patchelf
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco150/bin:/usr/lib/nvidia-418' >> ~/.bashrc
pip install mujoco_py
```
6. Install rl-baselines-zoo (the `sed` command changes a line in `train.py`)
```
git clone https://github.com/araffin/rl-baselines-zoo.git
mv rl-baselines-zoo rl_baselines_zoo
sed -n '131 c\    with open(os.path.join(os.path.dirname(__file__),"hyperparams/{}.yml".format(args.algo)), "r") as f:' rl_baselines_zoo/train2.txt
```

### Potential errors/solutions (largely with mujoco)
Error: `GLEW initalization error: Missing GL version`
Solution: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`

Error: `Qobject movetothread ...`
Solution:
```
pip list|grep opencv-python
pip uninstall opencv-python
pip install opencv-python==4.3.0.36
```

## Run the script!
python ALICE.py


### FYI
All this code is written with eager execution. If you train and load an expert through rl_baselines_zoo, that's not a problem, but if you train the expert with stablebaselines, that might be a challenge, and you might need to run the expert once to generate the cached demo trajs, and then another separate session to run IL on those cached demos.

The gym-CIL folder has the LQR environment, which you can try playing with if you want.
I haven't gotten it integrated yet since it's a continuous action space, but I'll get there soon. To install it:
cd gym-CIL && python -m pip install -e .
