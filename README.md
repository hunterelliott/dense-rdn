# dense-rdn
Dense reaction-diffusion network optimization code, accompanying the manuscript:

["Exploring Complex Dynamical Systems via Nonconvex Optimization"](https://arxiv.org/abs/2301.00923)

This has been tested with python 3.8.5 and tensorflow up to 2.5.0 (w/ and w/out GPU acceleration).

### Setup:

1. Clone the repo, where `{DENSE_RDN_PATH}` is a path to wherever you want to clone it e.g. `~/code/dense-rdn`

`git clone git@github.com:hunterelliott/dense-rdn.git {DENSE_RDN_PATH}`

2. Create a virtual environment and activate it. Conda has issues with some of the packages on e.g. macos so venv is simpler:

`python -m venv ~/dense-rdn-venv`

`source ~/dense-rdn-venv/bin/activate`

3. Pip install the requirements. If you want GPU support un-comment tensorflow-gpu in the requirements.txt and comment out regular tensorflow. 

`pip install -r {DENSE_RDN_PATH}/requirements.txt`

5. Make sure it's all working right by adding the repo to your python path and cd-ing into the repo to run one of the training scripts (I know, I know...).

`export PYTHONPATH=$PYTHONPATH:{DENSE_RDN_PATH}`

`cd {DENSE_RDN_PATH}/train/`

`python fixed_target_model_training_script.py`

6. You should now start seeing the log reporting training setup and eventually logging losses. You can use tensorboard to visualize the losses as well.

More README and general user-friendliness hopefully inbound soon... 
