

from config.default_config import*
import os 
from itertools import cycle, islice

import gzip

def default_params(width=None, height=None):
    params = DotDict()
    params.graph_mode = True
    params.batch_size = 4
    params.seq_len=100
    params.s_size= 64  # origin 45
    params.n_actions = 4

    # encoder and decoder 
    params.visual_dim = 64
    params.stoch_dim = 64
    
    # grid model
    params.p_size = 256
    params.g_size = 64
    params.g_size_project = params.p_size # g->P
    params.g_thresh_max = 10.0
    params.g_thresh_min = -10.0

    # sense model
    params.stoch_size = params.s_size  # 暂时假设enc-dec不做编码
    params.s_size_project = params.p_size # s->p
    params.s_size_hidden = 400

    # neural memory
    params.mem_dim = params.p_size  # g_size + s_size
    params.chunk_size = 1
    params.mem_batch_size = 1 
    params.heads = 1
    params.momentum_order = 2
    params.dim_head = 64
    params.learned_momentum_combine = False 

    # tem train process params
    params.temp_it = 2000
    params.p2g_start = -5  # 100  # iteration p2g kicks in
    params.p2g_warmup = 1  # 200
    params.g_gt_it = 20000000000  # 2000
    params.g_gt_bias = 0.0  # between -1 and 1
    params.g_reg_it = 20000000000

    # loss parameters
    params.which_costs = ['lx_g', 'lx_gt', 'lg']
    params.lx_gt_val = 1.0
    params.lg_val = 1.0
    params.lg_temp = 1.0
    params.l_encdec = 0.1

    # system 
    params.device = 'cuda:0'
    params.train_on_visited_states_only = True

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


def get_scaling_parameters(index, par):
    # these scale with number of gradient updates
    temp = np.maximum(np.minimum((index + 1) / par.temp_it, 1.0), 0.0)
    p2g_scale = 0.0 if index <= par.p2g_start else np.minimum((index - par.p2g_start) / par.p2g_warmup, 1.0)
    g_gt = par.g_gt_bias + np.maximum(np.minimum((index + 1) / par.g_gt_it, 1.0), -1.0)
    # g_cell_reg = 1 - np.minimum((index + 1) / par.g_reg_it, 1.0)

    scalings = DotDict({'temp': temp,
                   # 'l_r': l_r,
                   'iteration': index,
                   'p2g_scale': p2g_scale,
                   'g_gen': g_gt,
                   # 'g_cell_reg': g_cell_reg,
                   })

    return scalings