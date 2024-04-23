import os
from functools import partial

from .videogpt_reward_model import VideoGPTRewardModel#,UniPiRewardModel
from ... import CHECKPOINT_PATH

bind_videogpt_reward_model = lambda **kwargs: partial(VideoGPTRewardModel, **kwargs)
# bind_unipi_reward_model = lambda **kwargs: partial(UniPiRewardModel, **kwargs)

checkpoint_dir = os.getenv("VIPER_CHECKPOINT_DIR") or str(CHECKPOINT_PATH)
get_path = lambda dir: str(os.path.join(checkpoint_dir, dir))

LOAD_REWARD_MODEL_DICT = {
  # DeepMind Control Suite.
  'dmc_clen16_fskip1': bind_videogpt_reward_model(  # No frame skip
    videogpt_path=get_path('dmc_videogpt_l16_s1'),
    vqgan_path=get_path('dmc_vqgan')),
  'dmc_clen16_fskip2': bind_videogpt_reward_model(  # Frame skip 2
    videogpt_path=get_path('dmc_videogpt_l8_s2'),
    vqgan_path=get_path('dmc_vqgan')),
  'dmc_clen16_fskip4': bind_videogpt_reward_model(  # Frame skip 4
    videogpt_path=get_path('dmc_videogpt_l4_s4'),
    vqgan_path=get_path('dmc_vqgan')),
  'dmc_clen16_fskip4_cartpole_swingup': bind_videogpt_reward_model(  # Frame skip 4
    # videogpt_path='/data/zj/viper_rl/viper_rl_data/checkpoints/mine/dmc_videogpt_l4_s1_weight_weightvqgan',
    videogpt_path='/data/zj/viper_rl/viper_rl_data/checkpoints/mine/dmc_videogpt_l4_s1_cartpole_swingup_weight_weightvqgan',
    vqgan_path='/data/zj/viper_rl/viper_rl_data/checkpoints/mine/dmc_vqgan_weight'),
  'dmc_clen512_fskip128': bind_videogpt_reward_model(  # Frame skip 128
    videogpt_path=get_path('dmc_videogpt_l4_s128'),
    vqgan_path=get_path('dmc_vqgan')),
  # 'mine':bind_unipi_reward_model(videogpt_path=None,vqgan_path=get_path('dmc_vqgan')),

  # Atari.
  'atari_clen16_fskip1': bind_videogpt_reward_model(  # No frame skip
    videogpt_path=get_path('atari_videogpt_l16_s1'),
    vqgan_path=get_path('atari_vqgan')),
  'atari_clen16_fskip2': bind_videogpt_reward_model(  # Frame skip 2
    videogpt_path=get_path('atari_videogpt_l8_s2'),
    vqgan_path=get_path('atari_vqgan')),
  'atari_clen16_fskip4': bind_videogpt_reward_model(  # Frame skip 4
    videogpt_path=get_path('atari_videogpt_l4_s4'),
    vqgan_path=get_path('atari_vqgan')),
}