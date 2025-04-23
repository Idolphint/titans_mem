import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class GridModule(nn.Module):
    def __init__(self, action_dim, g_size, expand_factor=3):
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
        if not g0:
            g0 = self.g0.repeat(B, 1)
        new_g = self.norm_g(g0)  # 当前的Norm g可以进行多次而不改变结果
        grid_seq = []
        for t in range(S):
            new_g = self(new_g, act_seq[:, t, :])
            grid_seq.append(new_g)
        gen_g_seq = torch.stack(grid_seq, dim=1)
        return gen_g_seq

    # def gen_seq_g_tmp(self, pos_seq):
    #     pos_emb = self.tmp_emb(pos_seq)
    #     return pos_emb

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

        # loss = F.mse_loss(sense.detach(), pred)
        if return_dec:
            return loss, pred
        return loss


class MemNet(nn.Module):
    # 目前假设只有一层神经记忆，并且没有atten参与
    def __init__(self, dim, g_size, stoch_size):
        super().__init__()
        self.g2s_Memory = NeuralMemory(
            dim = dim,
            chunk_size=64,
            batch_size=128,  # 是根据batch_size将一个完整的seq分割成每个chunk 128的小段，所以这里叫batch_size不合理，
            qkv_receives_diff_views=True,
        )
        self.g_size = g_size
        self.stoch_size = stoch_size
        self.g2s_prev_weights = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def compute_loss(self,
                     g_seq,
                     stoch_seq,
                     cache=None,
                     return_cache=False):
        # 给定一个动作序列和视觉序列，要求可以根据动作预测出视觉
        # 这里的输入是grid和stoch，其中stoch已经是logit形式了
        # 计算根据动作驱动的g预测出s与真实s之间的kl散度，以及反过来的散度
        kld = torchd.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        value_seq = torch.cat((g_seq, stoch_seq), dim=-1) # 拼接后的结果
        g_key = torch.cat((g_seq, torch.zeros_like(stoch_seq)), dim=-1)  # 想象时用此查询

        # s1 学习根据g猜测s的能力，同时要保留g本身形态
        neural_mem_cache = cache
        kv_seq = torch.stack((g_key, g_key, value_seq))
        retrieved_sg_by_g, next_neural_mem_cache = self.g2s_Memory.forward(
            kv_seq,
            state = neural_mem_cache,
            prev_weights = None,  # 仅与layer有关，用于跨层残差更新
        )

        # 根据g和真实的s，学习到真实g的映射，期望这里可以学会纠错？这里的作用很奇怪啊？收敛的情况下期望这里学习到s,g到s,g的映射
        re_g_key = torch.cat((retrieved_sg_by_g[..., :self.g_size], stoch_seq), dim=-1)  # 取回g时用sense查询，TODO 设计可能需要修改
        re_kv_seq = torch.stack((re_g_key, re_g_key, value_seq))
        retrieved_sg_by_reg, next_neural_mem_cache2 = self.g2s_Memory.forward(
            re_kv_seq,
            state=next_neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )

        pred_stoch = retrieved_sg_by_g[..., self.g_size:]
        corr_g = retrieved_sg_by_reg[..., :self.g_size]
        query_seq = torch.cat((corr_g, torch.zeros_like(stoch_seq)), dim=-1)
        weights = next_neural_mem_cache2.updates
        retrived_stoch_by_reg = self.g2s_Memory.retrieve_memories(query_seq, weights)[..., self.g_size:]

        # loss1 pred_s -> stoch
        rep_loss = F.mse_loss(pred_stoch, stoch_seq.detach())
        retrived_rep_loss = F.mse_loss(retrived_stoch_by_reg, stoch_seq.detach())
        identity_loss = F.mse_loss(g_seq, corr_g)
        loss_detail = {'rep_loss': rep_loss,
                       "re_rep_loss": retrived_rep_loss,
                       "identity_loss": identity_loss,}
        # loss = rep_loss + retrived_rep_loss + identity_loss
        # self.sp2s(retrieved_s)
        # pred_dist = get_dist(pred_stoch)
        # gt_dist = get_dist(stoch_seq)
        #
        # rep_loss = kld(
        #     pred_dist, get_dist(stoch_seq.detach())
        # )
        # dyn_loss = kld(get_dist(pred_stoch.detach()), gt_dist)
        # rep_loss = F.mse_loss(pred_stoch, stoch_seq.detach())
        # dyn_loss = F.mse_loss(pred_stoch.detach(), stoch_seq)

        # loss = (1.0*rep_loss + 1.0*dyn_loss).mean()
        if return_cache:
            return loss_detail, next_neural_mem_cache2, pred_stoch
        return loss_detail, None,  pred_stoch

class Dynamic(nn.Module):
    def __init__(self, mem_dim, action_dim, visual_dim, g_dim, stoch_dim):
        super(Dynamic, self).__init__()
        self.encdec = SimpleEncDec(visual_dim, stoch_dim)
        self.grid_module = GridModule(action_dim, g_dim)
        self.mem = MemNet(mem_dim, g_dim, stoch_dim)

        self.mem_dim = mem_dim
        self.action_dim = action_dim
        self.visual_dim = visual_dim
        self.g_dim = g_dim
        self.stoch_dim = stoch_dim

    def forward(self, act_seq, visual_seq, mem_cache=None):
        # 假设都是 B,S,—
        B,S,_ = visual_seq.shape
        vis_enc_loss = self.encdec.compute_loss(visual_seq)
        gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        # gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        stoch_seq = self.encdec(visual_seq)

        dynamic_loss, mem_cache, pred_stoch = self.mem.compute_loss(gen_g_seq, stoch_seq, cache=mem_cache, return_cache=mem_cache is not None)
        imagine_dec_loss = self.encdec.compute_loss(visual_seq, pred_stoch)
        loss_detail = {"vis": vis_enc_loss.item(), "img": imagine_dec_loss.item()}
        loss = vis_enc_loss + imagine_dec_loss
        for k,v in dynamic_loss.items():
            loss_detail[k] = v.item()
            loss += v
        return loss, loss_detail

    def evaluate(self, act_seq, visual_seq, mem_cache=None):  # TODO mem不同的batch是否使用不同的mem呢？
        B,S,_ = visual_seq.shape
        dec_err, dec_sen = self.encdec.compute_loss(visual_seq, return_dec=True)
        # context_len1 = 128
        context_len = 64
        gen_g_seq = self.grid_module.gen_seq_g(act_seq)
        stoch_seq = self.encdec(visual_seq)

        # s1. 先把上下文记住
        dynamic_loss_0, mem_cache, _ = self.mem.compute_loss(gen_g_seq[:, :context_len], stoch_seq[:, :context_len], cache=None, return_cache=True)

        # s2. 再试图直接根据g取出内容
        query_seq = torch.cat((gen_g_seq, torch.zeros_like(stoch_seq)), dim=-1)[:, context_len:]
        weights = mem_cache.updates
        print(query_seq.shape, gen_g_seq.shape)
        pred_stoch = self.mem.g2s_Memory.retrieve_memories(query_seq, weights)[..., self.mem.g_size:]

        # dynamic_loss, mem_cache, pred_stoch = self.mem.compute_loss(gen_g_seq[:, context_len:], stoch_seq[:, context_len:], cache=mem_cache, return_cache=True)

        # img_err = self.encdec.compute_loss(visual_seq[:, context_len:], pred_stoch)
        img_sen_dec = self.encdec.dec(pred_stoch)
        visual_seq_label = torch.argmax(visual_seq, dim=-1)
        pred_label = torch.argmax(img_sen_dec, dim=-1)
        acc = (visual_seq_label[:, context_len:] == pred_label).float().mean()
        #
        # img_err = F.cross_entropy(img_sen_dec, visual_seq_label[:, context_len:].reshape(-1))
        print("pure dec err", dec_err, "img dec acc=", acc)
        pdb.set_trace()


class Trainer:
    def __init__(self):
        self.para = default_params()
        self.fix_train_env = init_env(self.para)
        self.test_env = init_env(self.para)
        self.dynamic = Dynamic(mem_dim=128, action_dim=self.para.env.n_actions,
                               visual_dim=self.para.s_size, g_dim=64, stoch_dim=64)
        self.opt = torch.optim.Adam(self.dynamic.parameters(), lr=1e-3)

    def train(self, num_episode=200):
        self.dynamic.cuda()
        self.dynamic.train()

        # env = self.fix_train_env  # 每个新数据更换一个环境，由于同时更换了视觉布局和记忆，所以期望能训练其一次记忆的能力

        for e in range(num_episode):
            env = init_env(self.para)
            traj_data = gen_data(env, self.para)
            pos = torch.from_numpy(traj_data["pos"]).cuda().long()
            act = torch.from_numpy(traj_data["dirs"]).cuda().float()  # a0是固定值，代表着到p0的动作
            sense = torch.from_numpy(traj_data["sense"]).cuda().float()

            loss, detail = self.dynamic(act, sense)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if e % 20 == 0:
                print(detail)
                torch.save(self.dynamic.state_dict(), "./saved_models/dynamic.pth")

    def eval(self):
        ckpt = torch.load("./saved_models/dynamic.pth")
        self.dynamic.load_state_dict(ckpt)
        self.dynamic.cuda()

        traj_data = gen_data(self.fix_train_env, self.para)
        pos = torch.from_numpy(traj_data["pos"]).cuda().long()
        act = torch.from_numpy(traj_data["dirs"]).cuda().float()  # a0是固定值，代表着到p0的动作
        sense = torch.from_numpy(traj_data["sense"]).cuda().float()
        with torch.no_grad():
            self.dynamic.evaluate(act, sense)

    @torch.no_grad()
    def visual(self):
        ckpt = torch.load("./saved_models/dynamic.pth")
        self.dynamic.load_state_dict(ckpt)
        self.dynamic.cuda()
        pos = torch.arange(0, self.fix_train_env.n_states).cuda().long()
        sense = sample_data(pos, self.fix_train_env.states_mat, s_size=self.para.s_size)
        raise NotImplementedError
        pos_emb = self.dynamic.grid_module.gen_seq_g(pos).cpu().numpy()
        # visual_emb = self.dynamic.encdec(sense)

        plt.figure(figsize=(10,6))
        sns.heatmap(pos_emb, cmap='coolwarm', xticklabels=4)
        plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    trainer = Trainer()
    # trainer.train(num_episode=10000)
    # trainer.visual()
    trainer.eval()
