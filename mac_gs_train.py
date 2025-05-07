import pdb
import random
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from envs.data_utils import *
from config.default_config import *
from GMS_net import FixGridModule
from adam_atan2_pytorch import AdoptAtan2
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention,
    DumyMemory,
)

# constants
ONLY_EVAL = True
DEVICE = "cuda:0"
NUM_BATCHES = int(1e5)
BATCH_SIZE = 8
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 64
GENERATE_LENGTH = 128
SHOULD_GENERATE = True
SEQ_LEN = 128

# neural memory related
MEM_TYPE = ""  # att, mlp, dumy
MODEL_DEPTH=8
NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = ()                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = MEM_TYPE == "att"
WINDOW_SIZE = 8
NEURAL_MEM_SEGMENT_LEN = 1                      # set smaller for more granularity for learning rate / momentum etc 为了更细粒度的lr和momentum
NEURAL_MEM_BATCH_SIZE = 32                     # set smaller to update the neural memory weights more often as it traverses the sequence  # 为了更贴近直接遍历seq, 这个的改变可能对训练影响有限
SLIDING_WINDOWS = True                     # 滑窗为False才能真的使用win_size的大小啊
STORE_ATTN_POOL_CHUNKS = NEURAL_MEM_SEGMENT_LEN>1                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS


# grid related
USE_GRID=False
GRID_DIM=32

# env related
params = env_param()
# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = True # turn this on to pipe experiment to cloud

# wandb experiment tracker
log_dir = (f'./saved_models/D{NEURAL_MEMORY_DEPTH}_Win{WINDOW_SIZE}{"+" if SLIDING_WINDOWS else ""}'
           f'_Seg{NEURAL_MEM_SEGMENT_LEN}_Nb{NEURAL_MEM_BATCH_SIZE}_G{GRID_DIM*USE_GRID}'
           f'{MEM_TYPE}_M{len(NEURAL_MEM_LAYERS)}_D{MODEL_DEPTH}_LM{NUM_LONGTERM_MEM+NUM_PERSIST_MEM}/')
writer = SummaryWriter(log_dir=log_dir if not ONLY_EVAL else None)
# import wandb
# wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
# wandb.run.name = RUN_NAME
# wandb.run.save()

# perf related

USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False

# memory model

if MEM_TYPE=="att":
    neural_memory_model = MemoryAttention(
        dim = 64
    )
elif MEM_TYPE=="dumy":
    neural_memory_model = DumyMemory(dim=0, depth=0)
else:
    print(MEM_TYPE)
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens = params.s_size,  # TODO 输入包含sense和act，需要自定义emb函数
    dim = 384,
    depth = MODEL_DEPTH,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
    ),
    token_emb="split",
    num_act=GRID_DIM if USE_GRID else params.n_actions,
    use_pos_emb=False,
).to(DEVICE)
if USE_GRID:
    grid_sys = FixGridModule(params.n_actions, g_size=GRID_DIM, device=DEVICE)

print(params.n_actions)
# optimizer
optim = AdoptAtan2([
    {"params": model.parameters(), "lr": LEARNING_RATE},
    # {"params": grid_sys.parameters(), "lr": LEARNING_RATE},
])

def train():
    model.train()
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # 数据准备
        env = init_env(params)
        traj_data = gen_data(env, params, SEQ_LEN, one_hot=False)
        if USE_GRID:
            # p_emb, g_emb = grid_sys.gen_seq_g_both(traj_data)
            g_emb = grid_sys.gen_seq_g(traj_data["pos"]).to(DEVICE)
            one_data = torch.cat([torch.from_numpy(traj_data["sense"]).to(DEVICE).unsqueeze(-1), g_emb], dim=-1).float()
        else:
            one_data = torch.from_numpy(np.stack((traj_data["sense"], traj_data["dirs"]), axis=-1)).to(DEVICE).long()
        loss_mask = torch.from_numpy(traj_data["visit"]).to(DEVICE).bool()
        loss = model(one_data, return_loss=True, loss_mask=loss_mask)
        # grid_loss = grid_sys.compute_loss(p_emb, g_emb, traj_data)
        # for name, l_ in grid_loss.items():
        #     loss += l_
        loss.backward()

    # print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    return loss.item()
    # wandb.log(log_loss)

def validate(detail=False):
    model.eval()
    with torch.no_grad():
        all_data = []
        all_mask = []
        env = init_env(params)
        traj_data = gen_data(env, params, SEQ_LEN, one_hot=False)
        if USE_GRID:
            # p_emb, g_emb = grid_sys.gen_seq_g_both(traj_data)
            g_emb = grid_sys.gen_seq_g(traj_data["pos"]).to(DEVICE)
            one_data = torch.cat([torch.from_numpy(traj_data["sense"]).to(DEVICE).unsqueeze(-1), g_emb], dim=-1).float()
        else:
            one_data = torch.from_numpy(np.stack((traj_data["sense"], traj_data["dirs"]), axis=-1)).to(DEVICE).long()
        loss_mask = torch.from_numpy(traj_data["visit"]).to(DEVICE).bool()


        print("valid ce ", model(one_data, return_loss=True, loss_mask=loss_mask))
        if detail:
            logits = model(one_data, return_loss=False)[:, :-1]
            prob = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(prob, dim=-1)
            labels = one_data[:, 1:, 0]
            select_prob = torch.gather(prob, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
            print("gt get prob=", select_prob.mean(), "pure=", 1/params.s_size)
            acc = (pred_label == labels).float().mean()
            seen_point_acc = ((pred_label == labels) * loss_mask[:, 1:]).sum() / loss_mask[:, 1:].sum()
            print("valid acc=", acc, seen_point_acc)

            if SLIDING_WINDOWS:
                B, S, _ = one_data.shape
                valid_win = (MODEL_DEPTH) * WINDOW_SIZE
                pos_equal = traj_data["pos"][:, np.newaxis] == traj_data["pos"][:, :, np.newaxis]
                pos_mask = np.zeros((B, S))
                for i in range(1, S):  # 获取在
                    pos_mask[:, i] = (~pos_equal[:, i, max(i - valid_win, 0):i].max(-1)) & (pos_equal[:, i, 0:i].max(-1))
                pos_mask = torch.from_numpy(pos_mask).to(DEVICE)[:,1:]
                out_win_acc = ((pred_label == labels) * pos_mask)
                print(out_win_acc.sum()/pos_mask.sum(), out_win_acc.sum(0)/pos_mask.sum(0))
                # ce_loss = F.cross_entropy(logits.transpose(1,2), labels, reduction='none') * loss_mask[:, 1:]
                # ce = ce_loss.sum() / loss_mask.sum()
                # print(f'validation loss: {ce.item()}')

def evaluate(detail=False):
    model.eval()
    test_env_n = 2 if detail else 1
    all_data = []
    all_mask = []
    all_pos = []
    for i in range(test_env_n):
        env = init_env(params)
        traj_data = gen_data(env, params, GENERATE_LENGTH, one_hot=False)
        if USE_GRID:
            # p_emb, g_emb = grid_sys.gen_seq_g_both(traj_data)
            g_emb = grid_sys.gen_seq_g(traj_data["pos"]).to(DEVICE)
            one_data = torch.cat([torch.from_numpy(traj_data["sense"]).to(DEVICE).unsqueeze(-1), g_emb], dim=-1).float()
        else:
            one_data = torch.from_numpy(np.stack((traj_data["sense"], traj_data["dirs"]), axis=-1)).to(DEVICE).long()
        loss_mask = torch.from_numpy(traj_data["visit"]).to(DEVICE).bool()
        all_mask.append(loss_mask)
        all_data.append(one_data)
        all_pos.append(traj_data["pos"])
    all_data = torch.cat(all_data, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    all_pos = np.concatenate(all_pos, axis=0)

    # 1-step acc
    logits = model(all_data, return_loss=False)[:, :-1]
    prob = F.softmax(logits, dim=-1)
    pred_label = torch.argmax(prob, dim=-1)
    labels = all_data[:, 1:, 0]
    select_prob = torch.gather(prob, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    print("gt get prob=", select_prob.mean(), "pure=", 1 / params.s_size)
    acc = (pred_label == labels).float().mean()
    seen_point_acc = ((pred_label == labels) * all_mask[:, 1:]).sum() / all_mask[:, 1:].sum()
    print("1-step acc=", acc, seen_point_acc)

    # multi-step acc
    B,S,_ = all_data.shape
    # batch_sample = torch.zeros((B,S-PRIME_LENGTH), device=DEVICE)
    inp = all_data[:, :PRIME_LENGTH]
    left_act = all_data[:, PRIME_LENGTH:, 1:].squeeze(-1)
    all_pred = model.sample(inp, GENERATE_LENGTH, use_cache=USE_FAST_INFERENCE, left_actions=left_act, temperature=0)  # 降低温度避免干扰

    acc = (all_data[:, PRIME_LENGTH:, 0] == all_pred).float()
    acc_ret = acc.mean()
    print("acc w/o mask=", acc_ret)
    # 检查前后完全一致的对
    if detail:
        before = all_pos[:, np.newaxis, :PRIME_LENGTH]
        after = all_pos[:, PRIME_LENGTH:, np.newaxis]
        has_same = np.max((before == after), axis=-1)  # B,after,before
        has_same = torch.from_numpy(has_same).to(DEVICE).bool()
        # pdb.set_trace()
        print("has same prob", has_same.float().mean(-1))
        all_pred[~has_same] = 0
        print("gt: ", all_data[:4, PRIME_LENGTH:, 0], "\npr: ", all_pred[:4])

        has_same = has_same.float()
        acc = (all_data[:, PRIME_LENGTH:, 0] == all_pred) * has_same.float()
        acc_with_step = acc.sum(0)/has_same.sum(0)
        acc_with_step = torch.nan_to_num(acc_with_step, nan=-0.05)
        t_range = np.arange(GENERATE_LENGTH-PRIME_LENGTH)
        plt.plot(t_range, acc.sum(0).cpu().numpy(), label='acc num ')
        plt.plot(t_range, has_same.sum(0).cpu().numpy(), label='has_same num')
        # plt.ylim(-0.1,1)
        plt.legend()
        plt.savefig(log_dir+f"/pred_with_mask_ctxt{PRIME_LENGTH}.jpg")
        # pdb.set_trace()
        acc_ret = acc.sum() / has_same.sum()
        print("acc with mask=", acc_ret, acc_with_step)

    return acc_ret.item()


if not ONLY_EVAL:
    last_save_acc = 0.0
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
        loss = train()
        writer.add_scalar("total", loss, i)
        if i % VALIDATE_EVERY == 0:
            validate()

        if SHOULD_GENERATE and i % GENERATE_EVERY == 0:  # 用于可视化输出的内容
            acc = evaluate()
            print("【gen】 decode acc=", acc)
            writer.add_scalar("gen_acc", acc, i)
            if acc > last_save_acc:
                torch.save(model.state_dict(), log_dir + "/mac_gs_best.pt")
                last_save_acc = acc
            torch.save(model.state_dict(), log_dir + "/mac_gs_latest.pt")
else:
    np.random.seed(101)
    torch.manual_seed(101)
    random.seed(101)
    log_dir = "/mnt/data/ltt/titans-pytorch/saved_models/D2_Win8+_Seg1_Nb32_G0_M0_D3/"
    print(log_dir + "/mac_gs_latest.pt")
    ckpt = torch.load(log_dir+"/mac_gs_latest.pt")
    model.load_state_dict(ckpt)
    validate(True)
    evaluate(True)
