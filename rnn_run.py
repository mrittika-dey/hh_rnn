import jax
from functools import partial
from jax import numpy as np
from jax import grad, jit, lax, random, vmap, jacfwd
from jax.nn import softmax
from jax.lax import scan
from inspect import signature
from jax.experimental import optimizers
from functools import reduce
from rnn_init import *
from utils import *

"""
Some of this code is from:

https://github.com/sussillo/computation-thru-dynamics/tree/master/integrator_rnn_tutorial
"""



def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)


def sigmoid(x):
    """ Implement   1 / ( 1 + exp( -x ) )   in terms of tanh."""
    return 0.5 * (np.tanh(x / 2.) + 1)


phis = {
  'ReLU':ReLU,
  'tanh':np.tanh,
  'linear':lambda x: x
}

def affine(params, x):
    """Implement y = w x """
    return np.dot(params['wO'], x)
    
batch_affine = vmap(affine,(None,0),0)

def def_rnn(phi,dt=0.1):

    def vanilla_rnn(params, h, x):
        """Run the Vanilla RNN one step"""
        cpar = np.concatenate([params['wI'],params['wR']],1)
        cI = np.concatenate([x,h],0)
        a = np.dot(cpar,cI) + params['bR']
        return phi(a)

    def cont_vanilla_rnn(params, h, x):
        """Run the Vanilla RNN one step"""
        dt = 0.1
        cpar = np.concatenate([params['wI'],params['wR']],1)
        cI = np.concatenate([x,h],0)
        a = np.dot(cpar,cI) + params['bR']
        h_dot = -h + phi(a)
        h_new = h + dt * h_dot
        return h_new

    def noisy_vanilla_rnn(params, h, x):
        """Run the Vanilla RNN one step"""
        cpar = np.concatenate([params['wI'],params['wR']],1)
        cI = np.concatenate([x,h],0)
        a = np.dot(cpar,cI) + params['bR']
        return phi(a)

    def gru_rnn(params, h, x): #hps
        """Implement the GRU equations.
        Arguments:
          params: dictionary of GRU parameters
          h: np array of  hidden state
          x: np array of input
          bfg: bias on forget gate (useful for learning if > 0.0)
        Returns:
          np array of hidden state after GRU update"""
      
        #_, bfg = hps
        hx = np.concatenate([h, x], axis=0)
        ru = np.dot(params['wRUHX'], hx) + params['bRU']

        r, u = np.split(ru, 2, axis=0)
        #print(ru.shape)


        #u = u + bfg
        r = sigmoid(r)
        u = sigmoid(u)
        rhx = np.concatenate([r * h, x])
        c = phi(np.dot(params['wCHX'], rhx) + params['bC'])
        return u * h + (1.0 - u) * c
    
    def cont_gru_rnn(params, h, x): #hps
        """Implement the GRU equations.
        Arguments:
          params: dictionary of GRU parameters
          h: np array of  hidden state
          x: np array of input
          bfg: bias on forget gate (useful for learning if > 0.0)
        Returns:
          np array of hidden state after GRU update"""
        dt = 0.1
        #_, bfg = hps
        hx = np.concatenate([h, x], axis=0)
        ru = np.dot(params['wRUHX'], hx) + params['bRU']
        print(ru.shape)
        r, u = np.split(ru, 2, axis=0)
        print(r.shape,u.shape)
        #u = u + bfg
        r = sigmoid(r)
        u = sigmoid(u)
        rhx = np.concatenate([r * h, x])
        c = phi(np.dot(params['wCHX'], rhx) + params['bC'])
        h_dot = -h + u * h + (1.0 - u) * c
        h_new = h + dt * h_dot
        return h_new

    return [vanilla_rnn,cont_vanilla_rnn,noisy_vanilla_rnn,gru_rnn,cont_gru_rnn]


def generate_rnn_dict(dt = 0.1):
    rnns = {}
    for phi_name in phis.keys():
        rnn_names = ['Vanilla','CTVanilla','NoisyVanilla','GRU','CTGRU']
        rnns[phi_name] = {}
        for rnn_name,rnn in zip(rnn_names,def_rnn(phis[phi_name],dt = dt)):
            rnns[phi_name][rnn_name] = rnn
    return rnns

def scanify(func):
    """Return the output twice for scan."""
    def new_func(*args):
        V = func(*args)
        return V,V
    return new_func

def runnify(rnn_func):
    """
    Obtain a function that runs the RNN many time-steps
    from a function that evaluates a single RNN step.
    """
    def rnn_func_run(params, x_t):
        """Run rnn_func T steps, where T is shape[0] of input."""
        h0 = np.zeros(params['wO'].shape[1])
        f = partial(scanify(rnn_func), params)
        _, h_t = lax.scan(f, h0, x_t)
        o_t = batch_affine(params, h_t)
        return h_t, o_t
    return rnn_func_run

### this version is faulty. Next one replaces it.
def noisify(rnn):
    """
    Accepts a RNN function: rnn(params,h,x) -> h that evaluates one timestep of RNN dynamics.
    Returns a new function noisy_rnn(params,h,x_) = rnn(params,h,x) + eta
    where x_ = [key,scale,x]
    and eta is a iid normally distributed random vector.
    
    """
    def noisy_rnn(params,h,x_):
        print("h,x shapes: ",h.shape,x_.shape)
        key,scale,x = x_[:2],x_[2:3],x_[3:]
        print("key,scale,x",key.shape,scale.shape,x.shape)
        print("key",key)
        h_tp1 = rnn(params,h,x)
        print("h_tp1 shape",h_tp1.shape)
        r = random.normal(key.astype(int),h_tp1.shape)
        print("scale shapes: ",scale.shape)
        r = scale * r
        print("r,scale shapes: ",r.shape,scale.shape)
        h_tp1 = h_tp1 + r
        return h_tp1
    return noisy_rnn

def noisify(rnn):
    """
    Accepts a RNN function: rnn(params,h,x) -> h that evaluates one timestep of RNN dynamics.
    Returns a new function noisy_rnn(params,h,x_) = rnn(params,h,x) + eta
    where x_ = [eta,x]
    and eta is a vector of the same shape as h.
    
    """

    def noisy_rnn(params,h,x_):
        n = h.shape[0]
        eta,x = x_[:n],x_[n:]
        h_tp1 = rnn(params,h,x)
        h_tp1 = h_tp1 + eta
        return h_tp1
    return noisy_rnn


def vrnn_scan(params, h, x):
    """Run the Vanilla RNN one step, returning (h ,h)."""  
    h = vrnn(params, h, x)
    return h, h

def vrnn_run(params, x_t):
    """Run the Vanilla RNN T steps, where T is shape[0] of input."""
    h = np.zeros(params['wO'].shape[1])
    f = partial(scanify(vrnn), params)
    _, h_t = lax.scan(f, h, x_t)
    o_t = batch_affine(params, h_t)
    return h_t, o_t

def out_onlyify(rnn_run):
    return lambda params,h_t: rnn_run(params,h_t)[1]

def hidden_onlyify(rnn_run):
    return lambda params,h_t: rnn_run(params,h_t)[0]

def softmaxify(rnn_run):
    return lambda params,h_t: softmax(rnn_run(params,h_t),axis=-1).swapaxes(-1,-2)[0:1].swapaxes(-1,-2)

def batchify_over_params(func,param_keys,indim=0,outdim=0,batchify_input_dims=None):
    """
    Takes a function 'func' which takes 'params' 
    as the first argument in addition to other 
    arguments. 

    Returns the vmapped function for a batch of params,
    wherein the other arguments remain the same,
    i.e like running a batch of rnns on the same inputs.
    """
    params_in_keys = dict(zip(param_keys,[indim for p in param_keys]))
    if batchify_input_dims is None:
        l = len(signature(func).parameters.keys())
        other_keys = [None]*(l-1)
    else:
        other_keys = batchify_input_dims
    in_keys = tuple([params_in_keys]+other_keys)
    return vmap(func,in_keys,outdim)

def batchify_over_params_nd(func,param_keys,nd,indim=0,outdim=0,batchify_input_dims=None):
    """
    batchify_input_dims: None or list of lists. 
    The list contains in 'other_keys' for each vmap operation.
    
    """
    for i in range(nd):
        batchify_input_dims_ = None if batchify_input_dims is None else batchify_input_dims[i]
        func=batchify_over_params(func,param_keys,indim=indim,outdim=outdim,batchify_input_dims=batchify_input_dims_)
    return func


def get_rnn_and_init_keys_funcs(
                 run_type = 'rnn',
                 arch = 'Vanilla',
                 phi_type  = 'tanh',
                 get_output = True,
                 get_hidden = False,
                 param_dims = 0,
                 noisified = False,
                 trial_dims = 0):
    """
    !!!!!!!!OLDER VERSION!!!!!!!!!
    
    Custom made RNN function to fit your needs.
    
    run_type: One of 'rnn' - runs only one step with inputs params,h,x
                     'run' - runs many timesteps with h(0):=0 inputs params,x_t
                     
    arch: RNN architecture. One of 'Vanilla','GRU','CTVanilla','noisyVanilla'
    
    phi_type: Non-linearity for RNN. Must be one of 'tanh' or 'ReLU'.
    
    trial_batchify_dim: vmaps over input trials.
    
    """
    vanilla_rnn,CTvanilla_rnn,noisy_rnn,gru_rnn = def_rnn(phis[phi_type])
    
    param_keys = {'Vanilla':vanilla_param_keys,
                 'GRU':gru_param_keys}
    
    secretly = {'Vanilla':'Vanilla',
                 'GRU':'GRU',
                 'CTVanilla':'Vanilla',
                 'noisyVanilla':'Vanilla'}
    
    rnn = {
        'Vanilla':vanilla_rnn,
        'GRU':gru_rnn,
        'CTVanilla':CTvanilla_rnn,
        'noisyVanilla':noisy_rnn
    }
    func = rnn[arch]
    
    if noisified:
        func = noisify(func)
    
    if run_type == 'run':
        func = runnify(func)
    elif run_type == 'rnn':
        pass
    else:
        raise Exception("'run_type' must be one of 'rnn' or 'run'.")
        
    nout = 2
    if (not get_output) and get_hidden:
        func = hidden_onlyify(func)
        nout = 1
    elif get_output and (not get_hidden):
        func = out_onlyify(func)
        nout = 1
    elif (not get_output) and (not get_hidden):
        raise Exception("""You probably do not want a function with no output.
        Please make at least one of (get_output, get_hidden) true.""")
        
    if not (param_dims==0):
        func = batchify_over_params_nd(func,param_keys[secretly[arch]],param_dims)
        
    for i in range(trial_dims):
        in_keys = {
            'rnn':(None,i,i),
            'run':(None,i)
        }
        out_keys = tuple([i+param_dims]*nout)
        func = vmap(func,in_keys[run_type],out_keys)
    
    init_func = get_init_func(arch=arch,batchify_dim=param_dims)
    
    return func,init_func,param_keys[secretly[arch]]

def get_rnn_and_init_keys_funcs(
                 run_type = 'rnn',
                 arch = 'Vanilla',
                 phi_type  = 'tanh',
                 get_output = True,
                 get_hidden = False,
                 param_dims = 0,
                 noisified = False,
                 trial_dims = 0):
    """
    ALSO OLDER VERSION WITHOUT co_batch
    
    Custom made RNN function to fit your needs.
    
    run_type: One of 'rnn' - runs only one step with inputs params,h,x
                     'run' - runs many timesteps with h(0):=0 inputs params,x_t
                     
    arch: RNN architecture. One of 'Vanilla','GRU','CTVanilla','noisyVanilla'
    
    phi_type: Non-linearity for RNN. Must be one of 'tanh' or 'ReLU'.
    
    trial_batchify_dim: vmaps over input trials.
    
    """
    vanilla_rnn,CTvanilla_rnn,noisy_rnn,gru_rnn = def_rnn(phis[phi_type])
    
    param_keys = {'Vanilla':vanilla_param_keys,
                 'GRU':gru_param_keys}
    
    secretly = {'Vanilla':'Vanilla',
                 'GRU':'GRU',
                 'CTVanilla':'Vanilla',
                 'CTGRU':'GRU',
                 'noisyVanilla':'Vanilla'}
    
    rnn = {
        'Vanilla':vanilla_rnn,
        'GRU':gru_rnn,
        'CTGRU':cont_gru_rnn,
        'CTVanilla':CTvanilla_rnn,
        'noisyVanilla':noisy_rnn
    }
    func = rnn[arch]
    
    if noisified:
        func = noisify(func)
    
    nout = 1
    if run_type == 'run':
        func = runnify(func)
        nout = 2
        if (not get_output) and get_hidden:
            func = hidden_onlyify(func)
            nout = 1
        elif get_output and (not get_hidden):
            func = out_onlyify(func)
            nout = 1
        elif (not get_output) and (not get_hidden):
            raise Exception("""You probably do not want a function with no output.
            Please make at least one of (get_output, get_hidden) true.""")
    elif run_type == 'rnn':
        pass
    else:
        raise Exception("'run_type' must be one of 'rnn' or 'run'.")
        
    if not (param_dims==0):
        func = batchify_over_params_nd(func,param_keys[secretly[arch]],param_dims)
        
    for i in range(trial_dims):
        in_keys = {
            'rnn':(None,i,i),
            'run':(None,i)
        }
        out_keys = tuple([i+param_dims]*nout) if nout>1 else i+param_dims
        func = vmap(func,in_keys[run_type],out_keys)
    
    init_func = get_init_func(arch=arch,batchify_dim=param_dims)
    
    return func,init_func,param_keys[secretly[arch]]


def get_rnn_and_init_keys_funcs(
                 run_type = 'rnn',
                 arch = 'Vanilla',
                 phi_type  = 'tanh',
                 get_output = True,
                 get_hidden = False,
                 param_dims = 0,
                 noisified = False,
                 trial_dims = 0,
                 co_batch = False):
    """
    Custom made RNN function to fit your needs.
    
    run_type: One of 'rnn' - runs only one step with inputs params,h,x
                     'run' - runs many timesteps with h(0):=0 inputs params,x_t
                     
    arch: RNN architecture. One of 'Vanilla','GRU','CTVanilla','noisyVanilla'
    
    phi_type: Non-linearity for RNN. Must be one of 'tanh' or 'ReLU'.
    
    trial_batchify_dim: vmaps over input trials.
    
    co_batch: input dims are batched along with param dims.
    
    """
    vanilla_rnn,CTvanilla_rnn,noisy_rnn,gru_rnn,cont_gru_rnn = def_rnn(phis[phi_type])
    
    param_keys = {'Vanilla':vanilla_param_keys,
                 'GRU':gru_param_keys}
    
    secretly = {'Vanilla':'Vanilla',
                 'GRU':'GRU',
                 'CTVanilla':'Vanilla',
                 'CTGRU':'GRU',
                 'noisyVanilla':'Vanilla'}
    
    rnn = {
        'Vanilla':vanilla_rnn,
        'GRU':gru_rnn,
        'CTGRU':cont_gru_rnn,
        'CTVanilla':CTvanilla_rnn,
        'noisyVanilla':noisy_rnn
    }

    
    in_keys = lambda i: {
        'rnn':(None,i,i),
        'run':(None,i)
    }

    
    func = rnn[arch]
    
    if noisified:
        func = noisify(func)
    
    nout = 1
    if run_type == 'run':
        func = runnify(func)
        nout = 2
        if (not get_output) and get_hidden:
            func = hidden_onlyify(func)
            nout = 1
        elif get_output and (not get_hidden):
            func = out_onlyify(func)
            nout = 1
        elif (not get_output) and (not get_hidden):
            raise Exception("""You probably do not want a function with no output.
            Please make at least one of (get_output, get_hidden) true.""")
    elif run_type == 'rnn':
        pass
    else:
        raise Exception("'run_type' must be one of 'rnn' or 'run'.")
        
    if not (param_dims==0):
        batchify_input_dims = [list((in_keys(i)[run_type]))[1:] for i in range(param_dims)] if co_batch else None
        print(batchify_input_dims)
        func = batchify_over_params_nd(func,param_keys[secretly[arch]],param_dims,batchify_input_dims=batchify_input_dims)
        
    for i in range(trial_dims):
        out_keys = tuple([i+param_dims]*nout) if nout>1 else i+param_dims
        j = i + param_dims if co_batch else i
        func = vmap(func,in_keys(j)[run_type],out_keys)
    
    init_func = get_init_func(arch=arch,batchify_dim=param_dims)
    
    return func,init_func,param_keys[secretly[arch]]




def get_loss_func(rnn_run,param_dims=0,batch_summify=False,trial_dims=0):
    def loss_fn(params,data):
        inputs,target,mask = data
        out = rnn_run(params,inputs)
        out = np.expand_dims(out,np.arange(param_dims))
        mask = np.expand_dims(mask,np.arange(param_dims))
        L = np.sum(((target - out)*mask),(-1,-2))**2/np.sum(mask**2,(-1,-2))
        return L

    for i in range(param_dims+trial_dims):
        if batch_summify:
            loss_fn = summify(loss_fn,axis=0)
    return loss_fn


def get_init_func(arch='Vanilla',batchify_dim=0):
    secretly = {'Vanilla':'Vanilla',
                 'GRU':'GRU',
                 'CTGRU':'GRU',
                 'CTVanilla':'Vanilla',
                 'noisyVanilla':'Vanilla'}
    param_keys = {'Vanilla':vanilla_param_keys,
                  'GRU':gru_param_keys}
    init_param_func = {'Vanilla':init_vanilla_rnn_params,
                       'GRU':init_gru_rnn_params}
    
    func = init_param_func[secretly[arch]]
    if not (batchify_dim==0):
        for _ in range(batchify_dim):
            func = batchify_param_init(func,param_keys[secretly[arch]])
    return func

