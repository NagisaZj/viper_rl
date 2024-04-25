# Video Prediction Models as Rewards for Reinforcement Learning

Code for VIPER (Video Predcition Rewards), a general algorithm which leverages video prediction models as priors for Reinforcement Learning.

<img src='https://github.com/Alescontrela/viper_rl/assets/13845012/b941627c-2ce1-49c3-9894-8d0b0e939462' width='100%'>

If you found this code useful, please reference it in your paper:

```
@article{escontrela2023viper,
  title={Video Prediction Models as Rewards for Reinforcement Learning},
  author={Alejandro Escontrela and Ademi Adeniji and Wilson Yan and Ajay Jain and Xue Bin Peng and Ken Goldberg and Youngwoon Lee and Danijar Hafner and Pieter Abbeel},
  journal={arXiv preprint arXiv:2305.14343},
  year={2023}
}
```

For more information:
- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## VIPER 🐍

VIPER leverages the next-frame log likelihoods of a pre-trained video prediction model as rewards for downstream reinforcement learning tasks. The method is flexible to the particular choice of video prediction model and reinforcement learning algorithm. The general method outline is shown below:

<img src='https://github.com/Alescontrela/viper_rl/assets/13845012/ee402c86-152a-4cdc-b728-8120d4b574dd' width='75%'>

This code release provides a reference VIPER implementation which uses [VideoGPT](https://arxiv.org/abs/2104.10157) as the video prediction model and [DreamerV3](https://arxiv.org/abs/2301.04104) as the reinforcement learning agorithm.

## Install:

Create a conda environment with Python 3.8:

```
conda create -n viper python=3.8
conda activate viper
```

Install [Jax][jax].

Install dependencies:
```
pip install -r requirements.txt
```

## Downloading Data

Download the DeepMind Control Suite expert dataset with the following command:

```
python -m viper_rl_data.download dataset dmc
```

and the Atari dataset with:

```
python -m viper_rl_data.download dataset atari
```

This will produce datasets in `<VIPER_INSTALL_PATH>/viper_rl_data/datasets/` which are used for training the video prediction model. The location of the datasets can be retrieved via the `viper_rl_data.VIPER_DATASET_PATH` variable.

## Downloading Checkpoints

Download the DeepMind Control Suite videogpt/vqgan checkpoints with:

```
python -m viper_rl_data.download checkpoint dmc
```

and the Atari checkpoint with:

```
python -m viper_rl_data.download checkpoint atari
```

This will produce video model checkpoints in `<VIPER_INSTALL_PATH>/viper_rl_data/checkpoints/`, which are used downstream for RL. The location of the checkpoints can be retrieved via the `viper_rl_data.VIPER_CHECKPOINT_PATH` variable.

## Video Model Training

Use the following command to first train a VQ-GAN:
```
CUDA_VISIBLE_DEVICES=0 python scripts/train_vqgan.py -o viper_rl_data/checkpoints/mine/dmc_vqgan_10per -c viper_rl/configs/vqgan/dmc.yaml
```

To train the VideoGPT, update `ae_ckpt` in `viper_rl/configs/dmc.yaml` to point to the VQGAN checkpoint, and then run:
```
CUDA_VISIBLE_DEVICES=1,2,4,5 python scripts/train_videogpt.py -o /data/zj/viper_rl/viper_rl_data/checkpoints/mine/dmc_videogpt_l4_s1_weight_weightvqgan -c viper_rl/configs/videogpt/dmc.yaml 

python viper_rl/train_inverse_model.py

CUDA_VISIBLE_DEVICES=2,3 python -m viper_rl.eval_inverse_model
```

## Policy training

Checkpoints for various models can be found in `viper_rl/videogpt/reward_models/__init__.py`. To use one of these video models during policy optimization, simply specify it with the `--reward_model` argument.  e.g.

```
CUDA_VISIBLE_DEVICES=1 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_balance --reward_model=dmc_clen16_fskip4 --logdir=~/logdir

CUDA_VISIBLE_DEVICES=5 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cheetah_run --reward_model=dmc_clen16_fskip4 --logdir=./logs/dmc_cheetah_run/viper_099
CUDA_VISIBLE_DEVICES=5 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_swingup --reward_model=dmc_clen16_fskip4_cartpole_swingup --logdir=./logs/dmc_cartpole_swingup/mine_bc_reward_01_inverseloss_0 --seed 5
CUDA_VISIBLE_DEVICES=3 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_swingup --reward_model=dmc_clen16_fskip1 --logdir=./logs/dmc_cartpole_swingup/mine_08_fskip1 --reward_model_batch_size 16
CUDA_VISIBLE_DEVICES=3 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_balance --reward_model=dmc_clen16_fskip4 --logdir=./logs/dmc_cartpole_balance/viper_099
 --horizon 333

CUDA_VISIBLE_DEVICES=0 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_quadruped_run --reward_model=dmc_clen16_fskip4 --logdir=./logs/dmc_quadruped_run/ori4 


CUDA_VISIBLE_DEVICES=5 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_quadruped_run --reward_model=dmc_clen16_fskip4 --logdir=./logs/dmc_quadruped_run/viper_reproduce2 --run.from_checkpoint /data/zj/viper_rl/logs/dmc_quadruped_run/viper/checkpoint.ckpt 

CUDA_VISIBLE_DEVICES=4,5 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_quadruped_run --reward_model=mine --logdir=./logs/dmc_quadruped_run/mine/debug5 --run.from_checkpoint /data/zj/viper_rl/logs/dmc_quadruped_run/viper_jointfalse/checkpoint.ckpt 

CUDA_VISIBLE_DEVICES=0 python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_quadruped_run --reward_model=dmc_clen16_fskip4 --logdir=./logs/dmc_quadruped_run/viper_load_to_ori3  --run.from_checkpoint /data/zj/viper_rl/logs/dmc_quadruped_run/viper_jointfalse/checkpoint.ckpt 
```

Custom checkpoint directories can be specified with the `$VIPER_CHECKPOINT_DIR` environment variable. The default checkpoint path is set to `viper_rl_data/checkpoints/`.

**Note**: For Atari, you will need to install [atari-py][ataripy] and follow the Atari 2600 VCS ROM install instructions.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
[paper]: https://arxiv.org/pdf/2305.14343.pdf
[website]: https://escontrela.me/viper
[tweet]: https://twitter.com/AleEscontrela/status/1661363555495710721?s=20
[ataripy]: https://github.com/openai/atari-py
