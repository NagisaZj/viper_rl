from typing import Any
import numpy as np
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn

from .transformer import Transformer

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
class VideoGPT(nn.Module):
    config: Any
    ae: Any

    def setup(self):
        self.shape = (self.config.seq_len, *self.ae.latent_shape(self.config.image_size))
        self.model = Transformer(
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed
        )

    @property
    def metrics(self):
        return ['loss']

    def __call__(self, embeddings, label=None, decode_step=None, training=False):
        if self.config.class_cond:
            assert label is not None, f"label is required for class conditioned model"

        L = np.prod(self.shape)
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        if self.config.class_cond:
            label = jax.nn.one_hot(label, num_classes=self.config.n_classes)
        else:
            label = None

        return self.model(
            embeddings,
            mask=mask,
            label=label,
            decode_step=decode_step,
            deterministic=not training,
        )

    def log_prob(self, embeddings, encodings, label=None, text=None, text_mask=None, training=False, reduce_sum=True):
        logits = self(embeddings, label=label, text=text, text_mask=text_mask, training=training)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        if self.config.class_cond:
            nll = nll.reshape(*nll.shape[:2], -1)
            nll = (nll.max(-1) * np.prod(encodings.shape[2:]) + nll.sum(-1)) / (2 * np.prod(encodings.shape[2:]))
        else:
            if reduce_sum:
                nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        return -nll

    def log_prob(self, embeddings, encodings, label=None, training=False, reduce_sum=True):
        logits = self(embeddings, label=label, training=training)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        return -nll

    def logits_to_embedding(self,logits,ae):
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        encoding = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        embedding = ae.lookup(encoding.astype(jnp.int32))
        return embedding
    
    def compute(self,true_embedding,imagine_embedding,label,training,ae):
        size = true_embedding.shape[0]
        embedding_all = jnp.concatenate([true_embedding,imagine_embedding],0)
        if label.shape[0]!=1:
            large_label = jnp.concatenate([label,label],0)
        else:
            large_label = label
        # print(embedding_all.shape,large_label.shape,true_embedding.shape)
        next_all = self(embedding_all, label=large_label, training=training)
        true_next = next_all[:size]
        imagined_next = next_all[size:]
        
        nll = optax.softmax_cross_entropy(true_next, jax.nn.softmax(imagined_next,-1))
        
        return nll,self.logits_to_embedding(true_next[:,-1],ae),self.logits_to_embedding(imagined_next[:,-1],ae)
    
    
    
    def my_reward3(self, embeddings, encodings, label=None, training=False, reduce_sum=True,ae=None):
        # embeddings = jnp.concatenate([embeddings[:,:1],embeddings[:,:-1]], axis=1)
        # encodings = jnp.concatenate([encodings[:,:1],encodings[:,:-1]], axis=1)
        logits = self(embeddings[:,:-1], label=label, training=training)
        step_embedding = self.logits_to_embedding(logits,ae)
        current_step_embedding = step_embedding[:,-1]
        
        print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        labels = jax.nn.one_hot(encodings[:,:-1], self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        
        # nll = ((current_step_embedding-embeddings[:,:-1])**2).mean(-1)
        
        imagined_sequence = jnp.concatenate([embeddings[:,1:-2],current_step_embedding[:,None],embeddings[:,-1:]], axis=1)
        
        
        num_iters = 0
        true_embedding = embeddings[:,1:]
        imagine_embedding = imagined_sequence
        for i in range(num_iters):
            tmp_nll, true_next, imagine_next = self.compute(true_embedding,imagine_embedding,label,training,ae)
            nll +=(tmp_nll*1)
            true_embedding = jnp.concatenate([true_embedding[:,1:-1],true_next[:,None],true_embedding[:,-1:]], axis=1)
            imagine_embedding = jnp.concatenate([imagine_embedding[:,1:-1],imagine_next[:,None],imagine_embedding[:,-1:]], axis=1)
            
        
        # true_next = self(embeddings[:,1:], label=label, training=training)
        # imagined_next = self(imagined_sequence, label=label, training=training)
        
        # # true_next_embedding = self.logits_to_embedding(true_next,ae)
        # # imagined_next_embedding = self.logits_to_embedding(imagined_next,ae)
        # # nll = nll+((true_next_embedding-imagined_next_embedding)**2).mean(-1)
        
        # nll = nll+optax.softmax_cross_entropy(true_next, jax.nn.softmax(imagined_next,-1))
        
        
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        
        # dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        # prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        
        # nll2 = optax.cross_entropy(probs, labels)
        # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        # nll = optax.softmax_cross_entropy(logits, labels)
        # print( nll2.mean(),nll.mean())
        # print(nll.shape)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction.astype(jnp.int32)#prediction
    
    def loss(self, embeddings, encodings, label=None, training=True):
        loss = -self.log_prob(
            embeddings, encodings, label, training=training
        ).mean() / np.prod(self.shape[1:])
        return dict(loss=loss)
