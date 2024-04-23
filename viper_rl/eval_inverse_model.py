import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data/zj/viper_rl/viper_rl')
from videogpt.models import load_videogpt, AE
import jax
import jax.numpy as jnp
import os
import dreamerv3.embodied.envs.dmc as dmc
from dreamerv3.embodied import wrappers

from train_inverse_model import InverseModel, PredictionModel
# from .. import sampler

def wrap_env(env):
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.FlattenTwoDimObs(env)
  env = wrappers.ExpandScalars(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env

class Evaler:
    def __init__(self,env,inverse_model,prediction_model):
        self.device = jax.devices()[0]
        self.torch_device = torch.device('cuda:0')
        # print(f'Reward model devices: {self.device}')
        self.ae = AE(path='/data/zj/viper_rl/viper_rl_data/checkpoints/dmc_vqgan', mode='jit')
        self.ae.ae_vars = jax.device_put(self.ae.ae_vars, self.device)
        self.env = env
        self.inverse_model = inverse_model.to(self.torch_device)
        self.prediction_model = prediction_model.to(self.torch_device)
        
    def eval(self):
      scores = []
      for i in range(10):
          score = 0
          actions={'action':np.zeros(12),'reset':np.ones(1)}
          obs = env.step(actions)
          encoding = self.ae.encode(obs['image'][None,None])#[0]  # batch,length
          embedding = self.ae.lookup(encoding)
          model_input = np.concatenate([embedding]*3,1)
          model_input_torch = torch.from_numpy(model_input).to(self.torch_device)
          prediction = self.prediction_model(model_input_torch)
          action = self.inverse_model(model_input_torch,prediction)[0].cpu().data.numpy()
          actions={'action':action,'reset':np.zeros(1)}
          done = False
          cnt=0
          while not obs['is_last']:
            cnt+=1
            obs = env.step(actions)
            score+=obs['reward']
            encoding = self.ae.encode(obs['image'][None,None])#[0]  # batch,length
            embedding = self.ae.lookup(encoding)
            model_input = np.concatenate([model_input[:,1:],embedding],1)
            model_input_torch = torch.from_numpy(model_input).to(self.torch_device)
            prediction = self.prediction_model(model_input_torch)
            action = self.inverse_model(model_input_torch,prediction)[0].cpu().data.numpy()
            actions={'action':action,'reset':np.zeros(1)}
          print(score,cnt)
      return scores
if __name__=='__main__':
    
    env = dmc.DMC('quadruped_run',size=(64, 64), repeat=2, camera=-1)
    env = wrap_env(env)
    print(env.act_space['action'].shape)
    inverse_model,prediction_model = InverseModel(64,env.act_space['action'].shape[0]),PredictionModel(64)
    inverse_model.load_state_dict(torch.load('/data/zj/viper_rl/viper_rl_data/checkpoints/inverse_model/quadruped_run/inverse_debug2.pth'))
    prediction_model.load_state_dict(torch.load('/data/zj/viper_rl/viper_rl_data/checkpoints/inverse_model/quadruped_run/prediction_debug.pth'))
    
    evaler = Evaler(env,inverse_model,prediction_model)
    scores=evaler.eval()
    print(scores)