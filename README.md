# Interactive Shaping of Granular Media Using Reinforcement Learning

<a href="https://humanoidsbonn.github.io/granular_rl/"><img alt="cover" src="./docs/images/cover.png" width="50%"/></a>

This repository contains the accompanying code for the paper "Interactive Shaping of Granular Media with Reinforcement Learning" by B. Kreis, M. Mosbach, A. Ripke, M. E. Ullah, S. Behnke, M. Bennewitz accepted to the IEEE-RAS International Conference on Humanoid Robots (Humanoids).

## Installation
Setup a Python 3.9 virtual environment e.g. "sand-gym". We use [Virtualenv](https://virtualenv.pypa.io/en/latest/) with [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). Alternatively, you can use [Conda](https://docs.conda.io/en/latest/).

To create the virtual environment with Virtualenvwrapper you have to run:
```
mkvirtualenv --python="/usr/bin/python3.9" sand-gym
```

If you are using Virtualenv with Virtualenvwrapper, make sure that you are in the correct environment.
```
workon sand-gym
```

Create a new root folder and enter it.
```
mkdir rl_sand_shaping
cd rl_sand_shaping
```

Then clone the repository and install the required requirements.
```
git clone https://github.com/benediktkreis/granular_rl.git
cd sand_gym
pip3 install --use-pep517 -r requirements.txt
cd ..
```

Install the robosuite:
```
git clone https://github.com/RoboticManipulation/robosuite.git
cd robosuite
pip3 install --use-pep517 -e .
python3 robosuite/scripts/setup_macros.py
cd ..
```

Go to the root folder of sand gym and install everything with ```pip install --use-pep517 -e .```
```
cd sand_gym
pip3 install --use-pep517 -e .
cd sand_gym
```

To see the GUI on OS >= Ubuntu 22, you have to add the following line to your ```.bashrc```
```
xhost +SI:localuser:$USER
```

## Usage
If you are using Virtualenv with Virtualenvwrapper, make sure that you are in the correct environment.
```
workon sand-gym
```

### Show available parameters
To see the available parameters check out the top of the corresponding Python file or print them using the -h argument:
```
python3 rl.py -h
```

### View a xml model
```
python3 -m mujoco.viewer --mjcf=model.xml
```

### Test an RL agent
The model name and the tensorboard log name are derived from the config file name. By default the GUI is not shown which allows to train in headless mode. If you want to see the GUI, you have to set the -g argument as shown below.

```
python3 rl.py -m=test -c=config_file_name -g=y
```

### Evaluate an RL agent
To evaluate an agent model, you have to specify the config file name. Upon execution of the command below, the model is pulled from Huggingface:

```
python3 rl.py -m=eval -c=config_file_name
```

Please note that the provided agents and goal height maps are examples and cannot reproduce the results of the paper.

### Train an RL agent
#### Training Script
```
python3 rl.py -m=train -c=config_file_name -g=y
```

#### Training Recommendations
##### Parallelization
Use the ```--num_cpu``` parameter to train with multiple parallel environments.

##### Specify training cores
```
taskset -c <cores> <script>
```
E.g. to run the Python scpript rl.py on cores 0-3:
```
taskset -c 0-3 python3 rl.py
```
This is the same as:
```
taskset -c 0,1,2,3 python3 rl.py
```

##### Specify training GPUs
E.g. to train on device 0:
```
export CUDA_VISIBLE_DEVICES=0
```
Alternatively, this can be added to the ```.bashrc```.

### Weights and Biases Support
Tensorboard logs are uploaded to Weights and Biases. If you use it for the first time, you have to login in and enter your credentials:
```
wandb login
```

### Hugging Face Hub Support
Models are stored locally and in Hugging Face Hub. If the model is not available locally, it will pull it from Hugging Face Hub. If you use Hugging Face for the first time, you have to generate a token for your machine [here](https://huggingface.co/settings/tokens)

Afterwards, you have to login once using the command line and enter your token. When prompted to change your git credentials, press "n":
```
huggingface-cli login
```

## Known issues
In case you are getting the error message ```"Failed to make the EGL context current."```, EGL has problems to do off-screen rendering, e.g., when working with image or pointclouds. To make it work, you can turn off the rendering on the GPU. For this purpose open:
```
/rl_sand_shaping/robosuite/robosuite/macros_private.py
```
and set:
```
MUJOCO_GPU_RENDERING = False
```

## Citation
Please cite our research as:
```
@inproceedings{kreis25humanoids,
  title={Interactive Shaping of Granular Media with Reinforcement Learning}, 
  author={B. Kreis, M. Mosbach, A. Ripke, M. E. Ullah, S. Behnke, M. Bennewitz},
  booktitle={Proc. of the IEEE-RAS Int. Conf. on Humanoid Robots (Humanoids)},
  year={2025}
}
```