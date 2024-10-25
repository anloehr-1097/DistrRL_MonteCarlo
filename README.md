# Introduction
This codebase is associated with my Master thesis titled "Policy Evaluation In Distributional Reinforcement Learning".
Although already functional, this is still a work in progress which may be updated in the future.

# Installation
Since PipEnv is used for package management, the installation requires the
exact Python version specified in the Pipfile.lock. This might need not
coincide with the Python version you have installed on your system. To
install the required Python version, you can use pyenv or conda.
Also, ensure that pipenv is installed on your system.
Here, the installation process with conda is described.

```bash
# create a new conda environment with python version (as of today: 3.10)
conda create -n <env_name> python=3.10
conda activate <env_name>
pipenv sync
```

# Usage
To run the experiments, take a look into `experiments.ipynb`.
To run the tests, simply run the following command from the root directory.
```bash
python -m unittest
```
*Note: 1 of the tests might fail due to numerical inaccuracies (test_grid_value_projection).*

# TODOS
## Extension of the DDP framework
The extended DDP algorithm should return two estimates in each iteration:
- One estimate for the return distribution function
- One estimate for the reward distribution collection
This is a simple change with the potential change to break the code.
Thus it is postponed to a later stage.

# Citation
If you use this code or the results of the paper in your research,
please cite it as follows:

[Info will be provided shortly]

**Note: Use of this code is permitted only with proper citation.**

