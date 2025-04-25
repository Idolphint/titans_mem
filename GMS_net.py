import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from hpc_memory_net import MemNet, BiMemNet
from titans_pytorch import NeuralMemory
import torch.distributions as torchd
from envs.data_utils import *
import matplotlib.pyplot as plt
import seaborn as sns


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


def get_dist(logits, class_num=32, unimix_ratio=0.01):
    logit_shape = logits.shape
    logits = logits.reshape((*logit_shape[:-1], class_num, logit_shape[-1]//class_num))
    dist = torchd.independent.Independent(
        OneHotDist(logits, unimix_ratio=unimix_ratio), 1
    )
    return dist


class FixGridModule():
    def __init__(self, action_dim, g_size, max_states=300, device='cuda:0'):
        self.module_size = [3, 7, 11]
        self.g_size = g_size
        self.max_states = max_states
        if g_size > sum(self.module_size):
            self.module_size.append(g_size-sum(self.module_size))
        self.g_book = torch.zeros((max_states, g_size)).to(self.device)
        ind = torch.arange(self.max_states).to(self.device)
        start_ind = 0
        for m in self.module_size:
            id_m = torch.fmod(ind, m)
            self.g_book[ind, id_m+start_ind] = 1.0
            start_ind += m
        print(self.g_book[:5, :10])

    def gen_seq_g(self, pos_seq):
        pos_shape = pos_seq.shape
        pos_emb = self.g_book[pos_seq.reshape(-1)]
        pos_emb = torch.reshape(pos_emb, (*pos_shape, self.g_size))
        # pos_emb = self.tmp_emb(pos_seq)
        return pos_emb


class GridModule(nn.Module):
    def __init__(self, action_dim, g_size, expand_factor=3, device='cuda:0'):
        super(GridModule, self).__init__()
        self.a2trans_layer = nn.Sequential(
            nn.Linear(action_dim, g_size*expand_factor),
            nn.ReLU(),
            nn.Linear(g_size*expand_factor, g_size*expand_factor),
            nn.ReLU(),
            nn.Linear(g_size*expand_factor, g_size**2),
            nn.ReLU(),
        )
        self.tmp_emb = nn.Embedding(300, g_size)
        self._init_weights()

        self.g0 = torch.nn.Parameter(
            torch.zeros(1, g_size), requires_grad=True
        )
        self.action_dim = action_dim
        self.g_size = g_size

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def norm_g(self, g):
        g = F.relu(g)
        return g / (torch.sum(g, dim=-1, keepdim=True) + 1e-8)

    def forward(self, prev_g, prev_a):
        g_shape = prev_g.shape
        W_trans = self.a2trans_layer(prev_a)
        W_trans = W_trans.reshape((*g_shape[:-1], self.g_size, self.g_size))
        new_g = torch.einsum('bgg,bg->bg', W_trans, prev_g)
        assert new_g.shape == g_shape
        new_g = self.norm_g(new_g)
        return new_g

    def gen_seq_g(self, act_seq, g0=None):
        B,S,_ = act_seq.shape
        if g0 is None:
            g0 = self.g0.repeat(B, 1)
        new_g = self.norm_g(g0)  # 当前的Norm g可以进行多次而不改变结果
        grid_seq = []
        for t in range(S):
            new_g = self(new_g, act_seq[:, t, :])
            grid_seq.append(new_g)
        gen_g_seq = torch.stack(grid_seq, dim=1)
        return gen_g_seq

    def gen_seq_g_tmp(self, pos_seq):
        pos_emb = self.tmp_emb(pos_seq)
        return pos_emb

class SimpleEncDec(nn.Module):
    def __init__(self, visual_dim, stoch_size):
        super(SimpleEncDec, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, stoch_size),
        )

        self.dec = nn.Sequential(
            nn.Linear(stoch_size, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, visual_dim),
        )

        self.visual_dim = visual_dim
        self.stoch_size = stoch_size

    def forward(self, sense, sample=True):
        logits = self.enc(sense)
        return logits
        # dist = get_dist(logits)
        # if sample:
        #     stoch = dist.sample()
        # else:
        #     stoch = dist.mode()
        # return stoch.reshape(logits.shape)

    def compute_loss(self, sense, stoch_logits=None, return_dec=False):  # decoder的训练可以考虑CE损失
        if stoch_logits is None:
            stoch = self(sense)
        else:
            # dist = get_dist(stoch_logits)
            # stoch = dist.sample()
            # stoch = stoch.reshape(stoch_logits.shape)
            stoch = stoch_logits
        pred = self.dec(stoch)
        sense_label = torch.argmax(sense, dim=-1).reshape(-1)
        pred_flatten = pred.reshape(-1, sense.shape[-1])
        loss = F.cross_entropy(pred_flatten, sense_label)
        pred_label = torch.argmax(pred, dim=-1).reshape(-1)
        acc = (pred_label==sense_label).float().mean()
        # loss = F.mse_loss(sense.detach(), pred)
        if return_dec:
            return loss, pred, acc
        return loss, acc


class Dynamic(nn.Module):
    def __init__(self, mem_dim, action_dim, visual_dim, g_dim, stoch_dim, para):
        super(Dynamic, self).__init__()
        self.para = para
        self.encdec = SimpleEncDec(visual_dim, stoch_dim)
        # self.grid_module = FixGridModule(action_dim, g_dim, 400, device=para.device)
        self.grid_module = GridModule(action_dim, g_dim, device=para.device)
        self.mem = BiMemNet(mem_dim, g_dim, stoch_dim, para.chunk_size, para.segment_size)

        self.mem_dim = mem_dim
        self.action_dim = action_dim
        self.visual_dim = visual_dim
        self.g_dim = g_dim
        self.stoch_dim = stoch_dim
        self.device = self.para.device

    def forward(self, act_seq, visual_seq, mem_cache=None):
        self.train()
        # 假设都是 B,S,—
        B,S,_ = visual_seq.shape
        vis_enc_loss, _ = self.encdec.compute_loss(visual_seq)
        gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        # gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        stoch_seq = self.encdec(visual_seq)

        dynamic_loss, mem_cache, pred_stoch = self.mem.compute_loss(gen_g_seq, stoch_seq, cache=mem_cache, return_cache=mem_cache is not None)
        imagine_dec_loss, acc_ = self.encdec.compute_loss(visual_seq, pred_stoch)
        loss_detail = {"vis": vis_enc_loss.item(), "img": imagine_dec_loss.item()}
        loss = vis_enc_loss + imagine_dec_loss
        for k,v in dynamic_loss.items():
            loss_detail[k] = v.item()
            loss += v
        return loss, loss_detail

    def seq_forward(self, act_seq, stoch_seq, last_g=None, cache=None):
        self.train()
        B,S,_ = act_seq.shape
        if last_g is None:
            last_g = self.grid_module.g0.repeat(B,1)

        gen_g_seq = self.grid_module.gen_seq_g(act_seq, last_g)

        pred_stoch, pred_stoch_corr, corr_g, next_cache = self.mem.step(gen_g_seq, stoch_seq, cache=cache,
                                                                    return_cache=True) # 记忆s-g双向，并返回预测的s和纠正的g

        return pred_stoch, pred_stoch_corr, gen_g_seq, corr_g, next_cache


    def evaluate2(self, act_seq, visual_seq, cache=None, context = 128):
        # 1. 首先记忆上下文，先根据g记忆s，并检查记忆效果，此时还要检查未来g取s的效果
        # 2. 接着根据gs记忆gs，检查以此为权重时的g取s效果和g_corr取s的效果
        # 3. 检查未来未corr的g直接取s的效果
        B, S, _ = visual_seq.shape
        dec_err, dec_sen, _ = self.encdec.compute_loss(visual_seq, return_dec=True)
        g_seq = self.grid_module.gen_seq_g(act_seq)
        stoch_seq = self.encdec(visual_seq)

        value_seq = torch.cat((g_seq, stoch_seq), dim=-1)  # 拼接后的结果
        g_key = torch.cat((g_seq, torch.zeros_like(stoch_seq)), dim=-1)  # 想象时用此查询

        neural_mem_cache = cache
        kv_seq_context = torch.stack((g_key[:, :context], g_key[:, :context], value_seq[:, :context]))
        sg_context_0, next_neural_mem_cache = self.mem.g2s_Memory.forward(  # 记忆g-s
            kv_seq_context,
            state=neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        # _, g2s_acc = encdec.xx(sg_context_0)  # 检查已知g取s的效果
        # 检查未知g取s的效果
        s_pred_future_0 = self.mem.g2s_Memory.retrieve_memories(g_key[:, context:], next_neural_mem_cache.updates)[...,
                          self.mem.g_size:]

        # s2 记忆sg->sg
        g_pred, s_pred = sg_context_0[..., :self.mem.g_size], sg_context_0[..., self.mem.g_size:]
        re_g_key = torch.cat((g_pred, stoch_seq[:, :context]), dim=-1)
        re_kv_seq = torch.stack((re_g_key, re_g_key, value_seq[:, :context]))
        sg_context_1, next_neural_mem_cache2 = self.mem.g2s_Memory.forward(
            re_kv_seq,
            state=next_neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        re_g_key = torch.cat((sg_context_1[..., :self.mem.g_size], torch.zeros_like(stoch_seq[:, :context])), dim=-1)
        s_corr_context = self.mem.g2s_Memory.retrieve_memories(re_g_key, next_neural_mem_cache2.updates)[...,
                          self.mem.g_size:]

        # sg_context_1 取s
        s_pred_future_1 = self.mem.g2s_Memory.retrieve_memories(g_key[:, context:], next_neural_mem_cache2.updates)[...,
                          self.mem.g_size:]

        _, acc0 = self.encdec.compute_loss(visual_seq[:, :context], sg_context_0[..., self.mem.g_size:])
        _, acc1 = self.encdec.compute_loss(visual_seq[:, context:], s_pred_future_0)
        _, acc2 = self.encdec.compute_loss(visual_seq[:, :context], sg_context_1[..., self.mem.g_size:])
        _, acc3 = self.encdec.compute_loss(visual_seq[:, context:], s_pred_future_1)
        _, acc4 = self.encdec.compute_loss(visual_seq[:, :context], s_corr_context)
        print("mem0 pred contxt", acc0, "pred future", acc1, "\n mem1 pred contxt", acc2, "pred future", acc3, acc4)

        # 下面是对训练的复现
        dynamic_loss, mem_cache, pred_stoch = self.mem.compute_loss(g_seq[:, context:], stoch_seq[:, context:], cache=None, return_cache=True)
        _, acc_ = self.encdec.compute_loss(visual_seq[:, context:], pred_stoch)
        pdb.set_trace()


    def evaluate(self, act_seq, visual_seq, mem_cache=None):  # TODO 不同batch会使用不同的mem
        B,S,_ = visual_seq.shape
        dec_err, dec_sen, _ = self.encdec.compute_loss(visual_seq, return_dec=True)
        # context_len1 = 128
        context_len = 128
        gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        stoch_seq = self.encdec(visual_seq)

        # s1. 先把上下文记住
        mem_cache = self.mem.compute_loss(
            gen_g_seq[:, :context_len],
            stoch_seq[:, :context_len],
            cache=None, return_cache=True, eval=True)

        # s2. 再试图直接根据g取出内容
        query_seq = torch.cat((gen_g_seq, torch.zeros_like(stoch_seq)), dim=-1)
        # weights = mem_cache.updates
        weights = mem_cache.states[0]  # 如果使用cache.updates那么保留了所有chunk单独的更新梯度，本质上没有混合seq信息
        for k, v in weights.items():  # 混合batch的权重使得测试时使用统一的权值
            weights[k] = torch.mean(v, dim=0, keepdim=True)
        # print(query_seq.shape, gen_g_seq.shape)
        pred_stoch = self.mem.g2s_Memory.retrieve_memories(query_seq, weights)[..., self.mem.g_size:]

        # dynamic_loss, mem_cache, pred_stoch = self.mem.compute_loss(gen_g_seq[:, context_len:], stoch_seq[:, context_len:], cache=mem_cache, return_cache=True)

        img_loss, img_acc = self.encdec.compute_loss(visual_seq[:, :context_len], pred_stoch[:, :context_len])
        print("pure dec err", dec_err, "img dec acc=", img_acc)
        # pdb.set_trace()


class Trainer:
    def __init__(self):
        self.para = default_params()
        self.device = self.para.device
        self.fix_train_env = init_env(self.para)
        self.test_env = init_env(self.para)
        self.dynamic = Dynamic(mem_dim=128, action_dim=self.para.env.n_actions,
                               visual_dim=self.para.s_size, g_dim=64, stoch_dim=64, para=self.para)
        self.opt = torch.optim.Adam(self.dynamic.parameters(), lr=1e-3)
        # self.schedule = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[1000, 2000, 6000], gamma=0.2)

    def train(self, num_episode=200):
        self.dynamic.to(self.device)
        self.dynamic.train()
        # env = self.fix_train_env  # 每个新数据更换一个环境，由于同时更换了视觉布局和记忆，所以期望能训练其一次记忆的能力

        # ckpt = torch.load("./saved_models/dynamic_nn.embG.pth")
        # print(self.dynamic.load_state_dict(ckpt, strict=False))

        for e in range(num_episode):
            env = init_env(self.para)
            traj_data = gen_data(env, self.para)
            pos = torch.from_numpy(traj_data["pos"]).to(self.device).long()
            act = torch.from_numpy(traj_data["dirs"]).to(self.device).float()  # a0是固定值，代表着到p0的动作
            sense = torch.from_numpy(traj_data["sense"]).to(self.device).float()

            loss, detail = self.dynamic(pos, sense)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # self.schedule.step()
            if e % 20 == 0:
                print(detail)
                torch.save(self.dynamic.state_dict(), f"./saved_models/dynamic_{self.para.save_prefix}.pth")
                # if e % 200 == 0:
                #     print("lr is ===", self.schedule.get_last_lr())
                    # self.eval(False)

    def seq_train(self, num_episode=200):
        self.dynamic.to(self.device)
        self.dynamic.train()
        for e in range(num_episode):
            env = init_env(self.para)
            traj_data = gen_data(env, self.para)
            pos = torch.from_numpy(traj_data["pos"]).to(self.device).long()
            act = torch.from_numpy(traj_data["dirs"]).to(self.device).float()
            sense = torch.from_numpy(traj_data["sense"]).to(self.device).float()
            stoch = self.dynamic.encdec(sense)
            B,S = pos.shape
            s = 0
            cache = None
            last_g = None
            seq_var = []
            while s < S:
                seg_pos = pos[:, s:s+self.para.segment_size]
                seg_act = act[:, s:s+self.para.segment_size]
                seq_stoch = stoch[:, s:s+self.para.segment_size]

                s += self.para.segment_size
                pred_s, pred_s_corr, g_gen, g_corr, cache= self.dynamic.seq_forward(seg_act, seq_stoch, last_g=last_g, cache=cache)  # 实现一个小batch的记忆更新与损失计算
                last_g = g_corr[:, -1]
                var_update = {'g_gen': g_gen, "g_corr": g_corr, 'pred_s': pred_s, 'pred_s_corr': pred_s_corr}
                seq_var.append(var_update)
            data = {}
            for k in seq_var[0].keys():
                data[k] = torch.cat([d[k] for d in seq_var], 1)

            # 计算损失1 积分的g与纠错的g一致
            g_corr_loss = F.mse_loss(data['g_gen'], data['g_corr'])

            # 积分的g和纠错的g都应当取出sense
            loss_pred, acc_pred = self.dynamic.encdec.compute_loss(sense, data['pred_s'])
            loss_corr, acc_corr = self.dynamic.encdec.compute_loss(sense, data['pred_s_corr'])

            loss = g_corr_loss + loss_pred + 0.001 * loss_corr

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if e % 20 == 0:
                print(f"{e}: g_loss={g_corr_loss.item():.2f}, pred_s={loss_pred.item():.2f}, corr_s={loss_corr.item():.2f}, acc_pred={acc_pred:.2f}, acc_corr={acc_corr:.2f}")


    def eval(self, load=True):
        if load:
            ckpt = torch.load("./saved_models/dynamic_store_once_and_retrive.pth")
            self.dynamic.load_state_dict(ckpt)
            self.dynamic.to(self.device)

        traj_data = gen_data(self.fix_train_env, self.para)
        pos = torch.from_numpy(traj_data["pos"]).to(self.device).long()
        act = torch.from_numpy(traj_data["dirs"]).to(self.device).float()  # a0是固定值，代表着到p0的动作
        sense = torch.from_numpy(traj_data["sense"]).to(self.device).float()
        with torch.no_grad():
            self.dynamic.evaluate(pos, sense)

    @torch.no_grad()
    def visual(self):
        ckpt = torch.load("./saved_models/dynamic.pth")
        self.dynamic.load_state_dict(ckpt)
        self.dynamic.to(self.device)
        pos = torch.arange(0, self.fix_train_env.n_states).to(self.device).long()
        sense = sample_data(pos, self.fix_train_env.states_mat, s_size=self.para.s_size)
        raise NotImplementedError
        pos_emb = self.dynamic.grid_module.gen_seq_g(pos).cpu().numpy()
        # visual_emb = self.dynamic.encdec(sense)

        plt.figure(figsize=(10,6))
        sns.heatmap(pos_emb, cmap='coolwarm', xticklabels=4)
        plt.show()


if __name__ == '__main__':
    seed = 101
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    trainer = Trainer()
    trainer.seq_train(num_episode=10000)
    # trainer.visual()
    trainer.eval()
