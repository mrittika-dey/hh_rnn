import jax
from functools import partial
from jax import numpy as np
from jax import grad, jit, lax, random, vmap, jacfwd
from jax.nn import softmax
from jax.lax import scan
from inspect import signature
from jax.experimental import optimizers
import matplotlib.pyplot as plt
from operator import mul
from functools import reduce


vanilla_param_keys = ['wI','wR','wO','bR']
def init_vanilla_rnn_params(key, u, n, o, g=1.0, scaling = "small"):
    """
    Generate vanilla RNN parameters
    Arguments:
      u,n,o: input,hidden,output sizes
      g: scaling of recurrent weights
    """

    key, *skeys = random.split(key, 4+1)
    skeys = iter(skeys)
    
    ifactor = 1.0 / np.sqrt(u)
    hfactor = g / np.sqrt(n)
    pfactor = 1.0 / {'small':n, 'large':np.sqrt(n)}[scaling]
    return {
            'wI' : random.normal(next(skeys), (n,u)) * ifactor,
            'wR' : random.normal(next(skeys), (n,n)) *  hfactor,
            'wO' : random.normal(next(skeys), (o,n)) * pfactor,
            'bR' : np.zeros([n])
            }

gru_param_keys = ['wRUHX','wCHX','bRU','bC','wO']
def init_gru_rnn_params(key, u, n, o,
                         i_factor=1.0, h_factor=1.0, pfactor=1.0, scaling = "small"):
    """
    Generate GRU parameters
    Arguments:
      key: random.PRNGKey for random bits
      n: hidden state size
      u: input size
      i_factor: scaling factor for input weights
      h_factor: scaling factor for hidden -> hidden weights
      h_scale: scale on h0 initial condition
    Returns:
      a dictionary of parameters
    """
    key, *skeys = random.split(key, 6+1)
    skeys = iter(skeys)
    
    ifactor = i_factor / np.sqrt(u)
    hfactor = h_factor / np.sqrt(n)
    
    wRUH = random.normal(next(skeys), (n+n,n)) * hfactor
    wRUX = random.normal(next(skeys), (n+n,u)) * ifactor
    wRUHX = np.concatenate([wRUH, wRUX], axis=1)
    
    wCH = random.normal(next(skeys), (n,n)) * hfactor
    wCX = random.normal(next(skeys), (n,u)) * ifactor
    wCHX = np.concatenate([wCH, wCX], axis=1)

    # Include the readout params in the GRU, though technically
    # not a part of the GRU.
    pfactor = pfactor / {'small':n, 'large':np.sqrt(n)}[scaling]
    wO = random.normal(next(skeys), (o,n)) * pfactor

    return {
            'wRUHX' : wRUHX,
            'wCHX' : wCHX,
            'bRU' : np.zeros((n+n,)),
            'bC' : np.zeros((n,)),
            'wO' : wO
            }

def batchify_param_init(init_func,param_keys,dim=0):
    out_keys = dict(zip(param_keys,[dim for _ in param_keys]))
    return vmap(init_func,(dim,None,None,None),out_keys)

vanilla_batch_init,gru_batch_init = map(batchify_param_init,[init_vanilla_rnn_params,init_gru_rnn_params],[vanilla_param_keys,gru_param_keys])