-- Requires python 3.6, and for baselines you should stick with tf=1.14

## Installing virtual environment (Python 3.6 on Ubuntu 16.04)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
sudo apt-get install python3.6-venv
python3.6 -m venv CIL_venv    ### or replace CIL_venv

## Activate virtual environment (venv)

source CIL_venv/bin/activate

(Type 'deactivate' to deactivate session)

## Install dependencies 
pip install --upgrade pip
pip install numpy matplotlib pandas setuptools>=49.1.0 pyqt5>=5.9.2 tensorflow==1.14 pytest cloudpickle~=1.2.0 gym[atari] sklearn

cd CIL_venv/lib && git clone https://github.com/openai/baselines.git && cd baselines && pip install -e . && cd ../../..
git clone https://github.com/araffin/rl-baselines-zoo.git && mv -r rl-baselines-zoo rl_baselines_zoo

## Stable-baselines
apt-get install swig cmake libopenmpi-dev zlib1g-dev ffmpeg
pip install requirements.txt
pip install stable-baselines[tests,docs]>=2.10.0 box2d-py==2.3.5 pybullet gym-minigrid scikit-optimize optuna pytablewriter seaborn pyyaml>=5.1

## Run the script!
python causal_IL.py

### FYI
All of my code is written to take advantage of eager execution since it's way
easier to debug what's going on. However, openAI baselines is written with
placeholders, so you can't use eager execution. My code defaults to running eager
and then turns it off when calling baselines, but if you try to do some other IL
stuff in the same session after calling baselines it'll run much slower. So, what
I typically do is train a baselines agent and save the model, then I have another
function that steals the matrices from the saved model and executes them eagerly

TLDR: If you want to train an expert policy on a new environment, train that agent
using the get_deepq_expert method (or alternatively act = deepq.learn(); act.save(path))
then run the script again once it's saved. Everything runs much more smoothly.

Also, all the environments thus far have been discrete, so model output is
probability simplex, and you need to take the argmax

The gym-CIL folder has the LQR environment, which you can try playing with if you want.
I haven't gotten it integrated yet since it's a continuous action space, but I'll
get there soon. To install it:
cd gym-CIL && python -m pip install -e .

Yeah, will probably have to take a vow of silence.
