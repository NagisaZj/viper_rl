import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from videogpt.models import load_videogpt, AE
import jax
import jax.numpy as jnp
import sys
sys.path.append('./')
import os

# from .. import sampler

class Trainer:
    def __init__(self,channels,action_dim):
        self.device = jax.devices()[0]
        self.torch_device = torch.device('cuda:0')
        # print(f'Reward model devices: {self.device}')
        self.ae = AE(path='/data/zj/viper_rl/viper_rl_data/checkpoints/dmc_vqgan', mode='jit')
        self.ae.ae_vars = jax.device_put(self.ae.ae_vars, self.device)
        self.inverse_model = InverseModel(channels,action_dim).to(self.torch_device)
        self.prediction_model = PredictionModel(channels).to(self.torch_device)
        self.inverse_optimizer = torch.optim.Adam(self.inverse_model.parameters(),3e-4)
        self.prediction_optimizer = torch.optim.Adam(self.prediction_model.parameters(),3e-4)
        
    def train(self,states,next_states,actions):
        # encodings = self.ae.encode(jnp.expand_dims(image_batch, axis=0))
        # embeddings = self.ae.lookup(encodings)
        states,next_states,actions = torch.from_numpy(states).to(self.torch_device),torch.from_numpy(next_states).to(self.torch_device), torch.from_numpy(actions).to(self.torch_device)
        # state_encodings = self.ae(jnp.expand_dims(states, axis=0))
        # next_state_encodings = self.ae(jnp.expand_dims(next_states, axis=0))
        # state_embeddings = self.ae.lookup(state_encodings)
        # next_state_embeddings = self.ae.lookup(next_state_encodings)
        
        predict_actions = self.inverse_model(states,next_states)
        loss = ((predict_actions-actions)**2).mean()
        
        self.inverse_optimizer.zero_grad()
        loss.backward()
        self.inverse_optimizer.step()
        
        predict_ns = self.prediction_model(states)
        # print(predict_ns.shape,next_states.shape,predict_actions.shape,actions.shape)
        loss2 = ((predict_ns-next_states)**2).mean()
        
        self.prediction_optimizer.zero_grad()
        loss2.backward()
        self.prediction_optimizer.step()
        
        return loss, loss2
        
    def prepare_data(self,states,actions):
        example_encoding = self.ae.encode(states[0:1,0:1])#[0]  # batch,length
        example_embedding = self.ae.lookup(example_encoding)
        # example_encoding,example_embedding = example_encoding[0],example_embedding[0]
        encodings = np.zeros([states.shape[0],states.shape[1],*example_encoding.shape[2:]],dtype=example_encoding.dtype)
        embeddings = np.zeros([states.shape[0],states.shape[1],*example_embedding.shape[2:]],dtype=example_embedding.dtype)
        # next_states = 
        batch_size = 16
        num_iters = states.shape[0] // batch_size
        # print(num_iters)
        for i in range(num_iters):
            encoding = self.ae.encode(states[i*batch_size:(i+1)*batch_size])
            embedding = self.ae.lookup(encoding)
            # encoding,embedding = encoding[0],embedding[0]
            
            
            encodings[i*batch_size:(i+1)*batch_size] = encoding
            embeddings[i*batch_size:(i+1)*batch_size] = embedding
        
        if (num_iters*batch_size)<states.shape[0]:
            encoding = self.ae.encode(states[num_iters*batch_size:])
            embedding = self.ae.lookup(encoding)
            # encoding,embedding = encoding[0],embedding[0]
            print(encodings.shape,encoding.shape,encodings[num_iters*batch_size:].shape)
            print(embeddings.shape,embedding.shape,embeddings[num_iters*batch_size:].shape)
            encodings[num_iters*batch_size:] = encoding
            embeddings[num_iters*batch_size:] = embedding
        
        # next_state_encodings, next_state_embeddings = 
        encodings,embeddings = encodings[:,:,None],embeddings[:,:,None]
        embedding_1 = np.concatenate([embeddings[:,0:1],embeddings[:,:-1]],1)
        embedding_2 = np.concatenate([embedding_1[:,0:1],embedding_1[:,:-1]],1)
        # embedding_1 = np.concatenate([embeddings[0:1],embeddings[:-1]],0)
        state_embeddings = np.concatenate([embedding_2,embedding_1,embeddings],2)[:,:-1]
        next_state_embeddings = embeddings[:,:-1]
        
        return state_embeddings.reshape(-1,*state_embeddings.shape[2:]),actions.reshape(-1,*actions.shape[2:]), next_state_embeddings.reshape(-1,*next_state_embeddings.shape[2:])
        

class InverseModel(nn.Module):
    def __init__(self, input_channels, output_size):
        super(InverseModel, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4 * 16 * 16 * 64, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, state_current, state_next):
        x = torch.cat((state_current, state_next), dim=1).permute(0,4,1,2,3)  # Concatenate along the channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print('I',x.shape)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PredictionModel(nn.Module):
    def __init__(self, input_channels):
        super(PredictionModel, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3,1,1), stride=1, padding=0)

    def forward(self, state_current):
        x = state_current.permute(0,4,1,2,3) # b,l,h,w,c->b,c,l,h,w
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # print('P',x.shape)
        x = self.conv4(x).permute(0,2,3,4,1)#.squeeze(1)
        # print(x.shape)
        return x

if __name__=='__main__':
    num_files = 300
    i=0
    all_states,all_actions = [],[]
    directory = '/data/zj/viper_rl/viper_rl_data/mydata/buffer/quadruped_run'
    for filename in os.listdir(directory):
        i+=1
        a=np.load(os.path.join(directory, filename),allow_pickle=True).tolist()
        states,actions = a['image'][None],a['action'][None]
        all_states.append(states)
        all_actions.append(actions)
        if i==num_files:
            break
    # print(states.shape,actions.shape,a['is_last'])
    trainer = Trainer(64,actions.shape[2])
    trainer.inverse_model.load_state_dict(torch.load('/data/zj/viper_rl/viper_rl_data/checkpoints/inverse_model/quadruped_run/inverse_debug2.pth'))
    state_embeddings, actions, next_state_embeddings = trainer.prepare_data(np.concatenate(all_states,0),np.concatenate(all_actions,0))
    print(state_embeddings.shape,next_state_embeddings.shape)
    # states,next_states,actions = 
    for i in range(10000):
        idx = np.random.choice(state_embeddings.shape[0],256)
        loss, loss2 = trainer.train(state_embeddings[idx],next_state_embeddings[idx],actions[idx])
        if i % 1000==0:
            print(i,loss, loss2)
            torch.save(trainer.inverse_model.state_dict(),'/data/zj/viper_rl/viper_rl_data/checkpoints/inverse_model/quadruped_run/inverse_debug2.pth')
            torch.save(trainer.prediction_model.state_dict(),'/data/zj/viper_rl/viper_rl_data/checkpoints/inverse_model/quadruped_run/prediction_debug2.pth')