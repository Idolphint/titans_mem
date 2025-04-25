from datetime import datetime

import numpy as np
from envs.environments import Rectangle


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data

def default_params(width=None, height=None):
    params = DotDict()
    params.graph_mode = True
    params.chunk_size = 32
    params.batch_size = 16
    params.segment_size = 64
    params.seq_len=256
    params.s_size=45
    params.world_type = 'rectangle'
    params.device = 'cuda:1'
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

