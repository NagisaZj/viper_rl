from typing import Any
import numpy as np
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn

from .transformer import Transformer


from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# from viper_rl.dreamerv3 import jaxutils
# from viper_rl.dreamerv3 import ninjax as nj

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

    def __call__(self, embeddings, label=None, decode_step=None, training=False,rightshift=True):
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
            rightshift=rightshift
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
        print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        # probs = jax.nn.softmax(logits,-1)
        # dist = tfd.Independent(jaxutils.OneHotDist(logits), 1)
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        # nll2 = optax.cross_entropy(probs, labels)
        # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        nll = optax.softmax_cross_entropy(logits, labels)
        # print( nll2.mean(),nll.mean())
        # print(nll.shape)
        # argmaxes = jnp.argmax(logits,-1)
        # prediction = jnp.abs(argmaxes-encodings)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        
        prediction = prediction.reshape(*prediction.shape[:2], -1).sum(-1)
        # print(nll.shape)
        # print(nll[:,-1])
        return -nll#, prediction
    
    def logits_to_embedding(self,logits,ae):
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        encoding = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        embedding = ae.lookup(encoding.astype(jnp.int32))
        return embedding
    
    def my_reward(self, embeddings, encodings, label=None, training=False, reduce_sum=True,ae=None):
        next_step_prediction_real = self(embeddings, label=label, training=training,rightshift=False)
        
        
        current_step_logits_all = self(embeddings, label=label, training=training)
        current_step_logits = current_step_logits_all[:,-1]
        # dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=current_step_logits)
        # currnet_step_encoding_prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        # current_step_embedding = ae.lookup(currnet_step_encoding_prediction.astype(jnp.int32))
        current_step_embedding = self.logits_to_embedding(current_step_logits,ae)
        imagine_embedding = jnp.concatenate([embeddings[:,:-1],current_step_embedding[:,None]], axis=1)
        next_step_prediction_imagine = self(imagine_embedding, label=label, training=training,rightshift=False)
        
        
        
        print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        # labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        # probs = jax.nn.softmax(logits,-1)
        # dist = tfd.Independent(jaxutils.OneHotDist(logits), 1)
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=current_step_logits_all)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        # nll2 = optax.cross_entropy(probs, labels)
        # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        # nll = optax.kl_divergence( jnp.log(jax.nn.softmax(next_step_prediction_real,-1)), jax.nn.softmax(next_step_prediction_imagine,-1))
        nll = optax.softmax_cross_entropy( next_step_prediction_real, jax.nn.softmax(next_step_prediction_imagine,-1))
        
        # next_step_real_embedding = self.logits_to_embedding(next_step_prediction_real[:,-1],ae)
        # next_step_imagine_embedding = self.logits_to_embedding(next_step_prediction_imagine[:,-1],ae)
        # real_sequence = jnp.concatenate([embeddings[:,1:],next_step_real_embedding[:,None]], axis=1)
        # imagine_sequence = jnp.concatenate([imagine_embedding[:,1:],next_step_imagine_embedding[:,None]], axis=1)
        
        # nn_real = self(real_sequence, label=label, training=training,rightshift=False)
        # nn_imagine = self(imagine_sequence, label=label, training=training,rightshift=False)
        # # nll+=optax.kl_divergence( jnp.log(jax.nn.softmax(nn_real,-1)), jax.nn.softmax(nn_imagine,-1))
        # nll+=optax.softmax_cross_entropy( nn_real, jax.nn.softmax(nn_imagine,-1))
        
        
        # print(nll.min())
        # print( nll2.mean(),nll.mean())
        # print(nll.shape)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction
    
    
    def my_reward2(self, embeddings, encodings, label=None, training=False, reduce_sum=True,ae=None):
        s_t_prediction = self(embeddings[:,:-1], label=label, training=training,rightshift=True)
        s_t_imagine_embedding = self.logits_to_embedding(s_t_prediction[:,-1],ae)
        imagine_embedding = jnp.concatenate([embeddings[:,1:-2],s_t_imagine_embedding[:,None],embeddings[:,-1:]], axis=1)
        imagine_prediction = self(imagine_embedding, label=label, training=training,rightshift=True)
        
        labels = jax.nn.one_hot(encodings[:,1:], self.ae.n_embed)
        nll = optax.softmax_cross_entropy( imagine_prediction, labels)
        
        ture_prediction_next = self(embeddings[:,1:], label=label, training=training,rightshift=False)
        s_t1_imagine_embedding = self.logits_to_embedding(imagine_prediction[:,-1],ae)
        imagine_embedding_next = jnp.concatenate([embeddings[:,1:-1],s_t1_imagine_embedding[:,None]], axis=1)
        imagine_prediction_next = self(imagine_embedding_next, label=label, training=training,rightshift=False)
        nll = nll + optax.softmax_cross_entropy( imagine_prediction_next, jax.nn.softmax(ture_prediction_next,-1))
        

        
        
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=s_t_prediction)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()

        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction
    
    
    
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




class MyVideoGPT(nn.Module):
    config: Any
    ae: Any

    def setup(self):
        self.shape = (self.config.seq_len, *self.ae.latent_shape(self.config.image_size))
        self.model = Transformer(
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed*5
        )

    @property
    def metrics(self):
        return ['loss']

    def __call__(self, embeddings, label=None, decode_step=None, training=False,rightshift=True):
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
            rightshift=rightshift
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
        # print(embeddings.shape,encodings.shape)
        logits = self(embeddings, label=label, training=training)
        # print(logits.shape)
        # print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        logits=logits.reshape(*encodings.shape,-1)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        # # probs = jax.nn.softmax(logits,-1)
        # # dist = tfd.Independent(jaxutils.OneHotDist(logits), 1)
        # dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        # prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        # prediction = prediction.reshape(*prediction.shape[:2], -1).sum(-1)
        # # nll2 = optax.cross_entropy(probs, labels)
        # # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        # # print( nll2.mean(),nll.mean())
        # # print(nll.shape)
        # # argmaxes = jnp.argmax(logits,-1)
        # # prediction = jnp.abs(argmaxes-encodings)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        
        
        # print(nll.shape)
        # print(nll[:,-1])
        return -nll#, prediction
    
    def logits_to_embedding(self,logits,ae):
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        encoding = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        embedding = ae.lookup(encoding.astype(jnp.int32))
        return embedding
    
    def my_reward(self, embeddings, encodings, label=None, training=False, reduce_sum=True,ae=None):
        next_step_prediction_real = self(embeddings, label=label, training=training,rightshift=False)
        
        
        current_step_logits_all = self(embeddings, label=label, training=training)
        current_step_logits = current_step_logits_all[:,-1]
        # dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=current_step_logits)
        # currnet_step_encoding_prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        # current_step_embedding = ae.lookup(currnet_step_encoding_prediction.astype(jnp.int32))
        current_step_embedding = self.logits_to_embedding(current_step_logits,ae)
        imagine_embedding = jnp.concatenate([embeddings[:,:-1],current_step_embedding[:,None]], axis=1)
        next_step_prediction_imagine = self(imagine_embedding, label=label, training=training,rightshift=False)
        
        
        
        print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        # labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        # probs = jax.nn.softmax(logits,-1)
        # dist = tfd.Independent(jaxutils.OneHotDist(logits), 1)
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=current_step_logits_all)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        # nll2 = optax.cross_entropy(probs, labels)
        # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        # nll = optax.kl_divergence( jnp.log(jax.nn.softmax(next_step_prediction_real,-1)), jax.nn.softmax(next_step_prediction_imagine,-1))
        nll = optax.softmax_cross_entropy( next_step_prediction_real, jax.nn.softmax(next_step_prediction_imagine,-1))
        
        # next_step_real_embedding = self.logits_to_embedding(next_step_prediction_real[:,-1],ae)
        # next_step_imagine_embedding = self.logits_to_embedding(next_step_prediction_imagine[:,-1],ae)
        # real_sequence = jnp.concatenate([embeddings[:,1:],next_step_real_embedding[:,None]], axis=1)
        # imagine_sequence = jnp.concatenate([imagine_embedding[:,1:],next_step_imagine_embedding[:,None]], axis=1)
        
        # nn_real = self(real_sequence, label=label, training=training,rightshift=False)
        # nn_imagine = self(imagine_sequence, label=label, training=training,rightshift=False)
        # # nll+=optax.kl_divergence( jnp.log(jax.nn.softmax(nn_real,-1)), jax.nn.softmax(nn_imagine,-1))
        # nll+=optax.softmax_cross_entropy( nn_real, jax.nn.softmax(nn_imagine,-1))
        
        
        # print(nll.min())
        # print( nll2.mean(),nll.mean())
        # print(nll.shape)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction
    
    
    def my_reward2(self, embeddings, encodings, label=None, training=False, reduce_sum=True,ae=None):
        s_t_prediction = self(embeddings[:,:-1], label=label, training=training,rightshift=True)
        s_t_imagine_embedding = self.logits_to_embedding(s_t_prediction[:,-1],ae)
        imagine_embedding = jnp.concatenate([embeddings[:,1:-2],s_t_imagine_embedding[:,None],embeddings[:,-1:]], axis=1)
        imagine_prediction = self(imagine_embedding, label=label, training=training,rightshift=True)
        
        labels = jax.nn.one_hot(encodings[:,1:], self.ae.n_embed)
        nll = optax.softmax_cross_entropy( imagine_prediction, labels)
        
        ture_prediction_next = self(embeddings[:,1:], label=label, training=training,rightshift=False)
        s_t1_imagine_embedding = self.logits_to_embedding(imagine_prediction[:,-1],ae)
        imagine_embedding_next = jnp.concatenate([embeddings[:,1:-1],s_t1_imagine_embedding[:,None]], axis=1)
        imagine_prediction_next = self(imagine_embedding_next, label=label, training=training,rightshift=False)
        nll = nll + optax.softmax_cross_entropy( imagine_prediction_next, jax.nn.softmax(ture_prediction_next,-1))
        

        
        
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=s_t_prediction)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()

        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction
    
    
    
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
        current_step_embedding = self.logits_to_embedding(logits[:,-1],ae)
        
        print('ae_embed',self.ae.n_embed,self.ae.ae.embed_dim,self.ae.ae.embed_dim)
        labels = jax.nn.one_hot(encodings[:,:-1], self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        
        # nll = ((current_step_embedding-embeddings[:,:-1])**2).mean(-1)
        
        imagined_sequence = jnp.concatenate([embeddings[:,1:-2],current_step_embedding[:,None],embeddings[:,-1:]], axis=1)
        
        
        num_iters = 3
        true_embedding = embeddings[:,1:]
        imagine_embedding = imagined_sequence
        for i in range(num_iters):
            tmp_nll, true_next, imagine_next = self.compute(true_embedding,imagine_embedding,label,training,ae)
            nll +=(tmp_nll*10)
            true_embedding = jnp.concatenate([true_embedding[:,1:-1],true_next[:,None],true_embedding[:,-1:]], axis=1)
            imagine_embedding = jnp.concatenate([imagine_embedding[:,1:-1],imagine_next[:,None],imagine_embedding[:,-1:]], axis=1)
            
        
        # true_next = self(embeddings[:,1:], label=label, training=training)
        # imagined_next = self(imagined_sequence, label=label, training=training)
        
        # # true_next_embedding = self.logits_to_embedding(true_next,ae)
        # # imagined_next_embedding = self.logits_to_embedding(imagined_next,ae)
        # # nll = nll+((true_next_embedding-imagined_next_embedding)**2).mean(-1)
        
        # nll = nll+optax.softmax_cross_entropy(true_next, jax.nn.softmax(imagined_next,-1))
        
        
        # dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        # prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))#nj.rng()
        dist = tfd.FiniteDiscrete(list(range(self.ae.n_embed)),logits=logits)
        prediction = dist.sample(seed=jax.random.PRNGKey(np.random.randint(10000000)))
        # nll2 = optax.cross_entropy(probs, labels)
        # print(logits.shape,encodings.shape,labels.shape)  # (32, 4, 16, 16, 256) (32, 4, 16, 16) (32, 4, 16, 16, 256)
        # nll = optax.softmax_cross_entropy(logits, labels)
        # print( nll2.mean(),nll.mean())
        # print(nll.shape)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        # print(nll.shape)
        return -nll, prediction

    def loss(self, embeddings, encodings, label=None, training=True):
        # print(embeddings.shape,encodings.shape)
        bs = embeddings.shape[1]//5
        loss = -self.log_prob(
            embeddings[:,:bs], encodings, label, training=training
        ).mean() / np.prod(self.shape[1:])
        return dict(loss=loss)
