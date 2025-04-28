import pdb

import numpy as np
from config.default_config import default_params
from envs.environments import *
from itertools import cycle, islice


def init_env(pars):
    if pars.world_type == 'rectangle':
        env = Rectangle(pars, pars.env.widths, pars.env.heights)
    if pars.world_type == 'hexagonal':
        env = Hexagonal(pars, pars.env.widths)
    if pars.world_type == 'family_tree':
        env = FamilyTree(pars, pars.env.widths)
    if pars.world_type == 'line_ti':
        env = LineTI(pars, pars.env.widths)
    if pars.world_type == 'wood2000':
        env = Wood2000(pars, pars.env.widths, pars.env.heights)
    if pars.world_type == 'frank2000':
        env = Frank2000(pars, pars.env.widths, pars.env.heights)
    if pars.world_type == 'grieves2016':
        env = Grieves2016(pars, pars.env.widths)
    if pars.world_type == 'sun2020':
        env = Sun2020(pars, pars.env.widths)
    if pars.world_type == 'nieh2021':
        env = Nieh2021(pars, pars.env.widths)

    env.world()
    env.state_data()
    env.walk_len = pars.seq_len  # 建议设置为>4*W*H的
        # batch_envs.append(env)

    return env


def sample_data(position, states_mat, s_size, one_hot=True):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    if one_hot:
        sense_data = np.zeros((s_size, time_steps))
    else:
        sense_data = np.zeros(time_steps)
    for i, pos in enumerate(position):
        ind = int(pos)
        if one_hot:
            sense_data[states_mat[ind], i] = 1
        else:
            sense_data[i] = states_mat[ind]
    return sense_data

def gen_data(env, pars, seq_len=0, one_hot=True):
    # 准备数据
    batch_pos = []
    batch_sense = []
    batch_dirs = []
    batch_envs = []

    for b in range(pars.batch_size):
        pos, direc = env.walk(rand_begin = False, seq_len=seq_len)
        sense = sample_data(pos, env.states_mat, s_size=pars.s_size, one_hot=one_hot)
        if not one_hot:
            direc = np.where(direc.sum(0)==0, 0, direc.argmax(axis=0)+1)
        batch_pos.append(pos)
        batch_sense.append(sense)
        batch_dirs.append(direc)
        batch_envs.append(env)
    if one_hot:
        data = {
            "pos": np.asarray(batch_pos),  # B,S
            "sense": np.asarray(batch_sense).transpose(0,2,1),  # B,S,_
            "dirs": np.asarray(batch_dirs).transpose(0,2,1),
        }
    else:
        data = {
            "pos": np.asarray(batch_pos),  # B,S
            "sense": np.asarray(batch_sense),
            "dirs": np.asarray(batch_dirs)
        }
    return data

# def prepare_inputs(env, pars):
#     for b in range(pars.batch_size):
#         pos, direc = env.walk()



if __name__ == '__main__':
    pars = default_params()
    env = init_env(pars)
    data = gen_data(env, pars)
    data = gen_data(env, pars)