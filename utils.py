import jax
from functools import partial
from jax import numpy as np
from jax import jit, random, vmap
from jax.nn import softmax
from jax.lax import scan
from inspect import signature
from operator import mul
from functools import reduce

def genkeys_array(key,n):
    """
    Get an iterator of n new keys and replace the old key.
    """
    new_keys = random.split(key,n+1)
    return new_keys[0],new_keys[1:]

def genkeys(key,n):
    """
    Get an iterator of n new keys and replace the old key.
    """
    new_keys = random.split(key,n+1)
    return new_keys[0],iter(new_keys[1:])

def add_gaussian_noise(key,x,sig=0.1):
    return random.normal(key,x.shape)*sig + x

def split_rnns(params,bs=100):
    all_params = []
    n = list(params.values())[0].shape[0]
    for i in range(0,n,bs):
        new_batch = {}
        for key in params.keys():
            new_batch[key] = params[key][i:i+bs]
        all_params.append(new_batch)
    return all_params

def append_noise_to_input(key,inputs,n,scale):
    inputs = np.concatenate(
        [scale*random.normal(key,(*inputs.shape[:-1],n)),
        inputs],-1)
    return inputs

def summify(func,axis=0,aux=False):
    """
    Arguments
    func: callable F(a,b,c,..) -> d
    axis: int
    aux: bool
    Returns: callable F_(a,b,c,...) = F(a,b,c,...).sum(axis)
     or F_(a,b,c,...) = F(a,b,c,...).sum(axis), F(a,b,c,...) 
    """
    if aux:
        def new_func(*args):
            F = func(*args)
            return F.sum(axis),F
    else:
        def new_func(*args):
            F = func(*args)
            return F.sum(axis)
    return new_func