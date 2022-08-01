import jax.numpy as np
import numpy as NP
from jax.nn import one_hot,sigmoid
from functools import partial
from jax import random,vmap,value_and_grad
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from rnn_init import *
from rnn_run import *
from utils import *


run,init = get_rnn_and_init_funcs(run_type='run',batchify_dim=0)

key = random.PRNGKey(0)
key,skey = random.split(key)

params = init(skey,1,10,1)
inp = np.ones((20,1))
out = run(params,inp)

print("Done.")