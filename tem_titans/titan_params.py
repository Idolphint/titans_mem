

from config.default_config import*
import os 
from itertools import cycle, islice

import gzip

def default_params(width=None, height=None):
    params = DotDict()
    params.graph_mode = True
    params.batch_size = 16
    params.seq_len=256
    params.s_size= 64  # origin 45
    params.n_actions = 4

    # encoder and decoder 
    params.visual_dim = 64
    params.stoch_dim = 64
    
    # grid model 
    params.g_size = 128
    params.g_size_project = params.g_size
    params.g_thresh_max = 10.0
    params.g_thresh_min = -10.0

    # sense model 
    params.s_size_project = params.s_size
    params.s_size_hidden = 400

    # neural memory
    params.mem_dim = params.g_size + params.s_size  # g_size + s_size
    params.chunk_size = 1
    params.mem_batch_size = 1 
    params.heads = 1
    params.momentum_order = 2
    params.dim_head = params.dim 
    params.learned_momentum_combine = False 
    
    
    # system 
    params.device = 'cuda:0'

    params.world_type = 'rectangle'
    params.rand_begin = False
    params.save_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    par_env = DotDict({'stay_still': True,
                  'bias_type': 'angle',
                  'direc_bias': 0.25,
                  'angle_bias_change': 0.4,
                  'restart_max': 40,
                  'restart_min': 5,
                  'seq_jitter': 30,
                  'save_walk': 30,
                  'sum_inf_walk': 30,
                  'widths': 11,  #[10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not width else [width] * params.batch_size,
                  'heights': 11, # [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not height else [height] * params.batch_size,
                  'rels': ['down', 'up', 'left', 'right', 'stay still'],
                  })
    size_increase = 6
    par_env.widths += size_increase  #[width + size_increase for width in par_env.widths]
    par_env.heights += size_increase  # [height + size_increase for height in par_env.heights]
    n_states = Rectangle.get_n_states(par_env.widths, par_env.heights)  # [Rectangle.get_n_states(width, height) for width, height in zip(par_env.widths, par_env.heights)]

    # repeat widths and height
    # par_env.widths = list(islice(cycle(par_env.widths), params.batch_size))
    # par_env.heights = list(islice(cycle(par_env.heights), params.batch_size))
    params.max_states = np.max(n_states)
    par_env.n_actions = len(par_env.rels) if 'stay still' not in par_env.rels else len(par_env.rels) - 1
    params.n_actions = par_env.n_actions
    params.env = par_env

    params.use_reward = True
    return params