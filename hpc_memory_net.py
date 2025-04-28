import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch import NeuralMemory


class MemNet(nn.Module):
    # 目前假设只有一层神经记忆，并且没有atten参与
    def __init__(self, dim, g_size, stoch_size, chunk_size=64, segment_size=128):
        super().__init__()
        self.g2s_Memory = NeuralMemory(
            dim = dim,
            chunk_size=chunk_size,
            batch_size=segment_size,  # 是根据batch_size将一个完整的seq分割成每个chunk 128的小段，所以这里叫batch_size不合理，
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

    def compute_loss_old(self, g_seq, stoch_seq, cache=None, return_cache=False):
        kv_seq = torch.stack((g_seq, g_seq, stoch_seq))
        pred_stoch, next_neural_mem_cache = self.g2s_Memory.forward(
            kv_seq,
            state=cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        rep_loss = F.mse_loss(pred_stoch, stoch_seq.detach())
        dyn_loss = F.mse_loss(pred_stoch.detach(), stoch_seq)
        loss_detail = {'rep': rep_loss, 'dyn': dyn_loss}  # 这里的损失只做可视化，不可用于训练，会导致
        # loss = (1.0*rep_loss + 1.0*dyn_loss).mean()
        if return_cache:
            return loss_detail, next_neural_mem_cache, pred_stoch
        return loss_detail, None, pred_stoch

    def evaluate(self, g_seq, stoch_seq, cache=None, context=128):
        # 1. 首先记忆上下文，先根据g记忆s，并检查记忆效果，此时还要检查未来g取s的效果
        # 2. 接着根据gs记忆gs，检查以此为权重时的g取s效果和g_corr取s的效果
        # 3. 检查未来未corr的g直接取s的效果

        value_seq = torch.cat((g_seq, stoch_seq), dim=-1)  # 拼接后的结果
        g_key = torch.cat((g_seq, torch.zeros_like(stoch_seq)), dim=-1)  # 想象时用此查询

        neural_mem_cache = cache
        kv_seq_context = torch.stack((g_key[:, :context], g_key[:, :context], value_seq[:, :context]))
        sg_context_0, next_neural_mem_cache = self.g2s_Memory.forward(  # 记忆g-s
            kv_seq_context,
            state=neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        # _, g2s_acc = encdec.xx(sg_context_0)  # 检查已知g取s的效果
        # 检查未知g取s的效果
        s_pred_future_0 = self.g2s_Memory.retrieve_memories(g_key[:, context:], next_neural_mem_cache.updates)[..., self.g_size:]

        # s2 记忆sg->sg
        g_pred, s_pred = sg_context_0[..., :self.g_size], sg_context_0[..., self.g_size:]
        re_g_key = torch.cat((g_pred, stoch_seq[:, :context]), dim=-1)
        re_kv_seq = torch.stack((re_g_key, re_g_key, value_seq[:, :context]))
        sg_context_1, next_neural_mem_cache2 = self.g2s_Memory.forward(
            re_kv_seq,
            state=next_neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        # sg_context_1 取s
        s_pred_future_1 = self.g2s_Memory.retrieve_memories(g_key[:, context:], next_neural_mem_cache2.updates)[..., self.g_size:]
        return sg_context_0[..., self.g_size:], s_pred_future_0, sg_context_1[..., self.g_size:], s_pred_future_1

    def compute_loss_old_2(self,
                     g_seq,
                     stoch_seq,
                     cache=None,
                     return_cache=False):
        # 给定一个动作序列和视觉序列，要求可以根据动作预测出视觉
        # 这里的输入是grid和stoch，其中stoch已经是logit形式了
        # 计算根据动作驱动的g预测出s与真实s之间的kl散度，以及反过来的散度

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
        g_pred, s_pred = retrieved_sg_by_g[..., :self.g_size], retrieved_sg_by_g[..., self.g_size:]
        # s2 根据g和真实的s，学习到真实g的映射，期望这里可以学会纠错？这里的作用很奇怪啊？收敛的情况下期望这里学习到s,g到s,g的映射
        re_g_key = torch.cat((g_pred, stoch_seq), dim=-1)  # 取回g时用sense查询，TODO 设计可能需要修改
        re_kv_seq = torch.stack((re_g_key, re_g_key, value_seq))
        retrieved_sg_by_reg, next_neural_mem_cache2 = self.g2s_Memory.forward(
            re_kv_seq,
            state=next_neural_mem_cache,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )

        corr_g = retrieved_sg_by_reg[..., :self.g_size]
        query_seq = torch.cat((corr_g, torch.zeros_like(stoch_seq)), dim=-1)
        weights = next_neural_mem_cache2.updates
        s_pred_pred = self.g2s_Memory.retrieve_memories(query_seq, weights)[..., self.g_size:]

        # loss1 pred_s -> stoch
        dyn_loss = F.mse_loss(s_pred, stoch_seq.detach())
        # rep_loss = F.mse_loss(s_pred.detach(), stoch_seq)  # 反过来促进stoch的表达更合理？或许是可有可无的
        retrived_rep_loss = F.mse_loss(s_pred_pred, stoch_seq.detach())
        identity_loss = F.mse_loss(g_seq, corr_g)
        loss_detail = {'dyn_loss': dyn_loss,
                       # 'rep_loss': rep_loss,
                       "re_rep_loss": retrived_rep_loss,
                       "identity_loss": identity_loss,}
        if return_cache:
            return loss_detail, next_neural_mem_cache2, s_pred_pred
        return loss_detail, None,  s_pred_pred

    def compute_loss(self,
                     g_seq,
                     stoch_seq,
                     cache=None,
                     return_cache=False,
                     eval=False):
        value_seq = torch.cat((g_seq, stoch_seq), dim=-1)  # 学习的目标
        g_key_seq = torch.cat((g_seq, torch.zeros_like(stoch_seq)), dim=-1)
        gs_key_seq = torch.cat((g_seq, stoch_seq), dim=-1)
        gs_key_seq_with_noise = torch.cat((g_seq + torch.randn_like(g_seq) * 0.1, stoch_seq), dim=-1)
        # kv_seq = torch.stack((g_key_seq, g_key_seq, value_seq))
        # kv_equal_seq = torch.stack((gs_key_seq_with_noise, gs_key_seq_with_noise, value_seq))

        # 记住从g->s以及从有噪音的g中取出g
        key_cat_seq = torch.cat((g_key_seq, gs_key_seq_with_noise), dim=1)
        value_cat_seq = value_seq.repeat((1, 2, 1))
        kv_seq_full = torch.stack((key_cat_seq, key_cat_seq, value_cat_seq))
        _, next_cache = self.g2s_Memory(kv_seq_full, state=cache, prev_weights=None, )
        # self.g2s_Memory.store(kv_seq)
        # self.g2s_Memory.store(kv_equal_seq)
        if return_cache and eval:
            return next_cache

        # 期望g可以取出s；g+noise,s可以取出g
        weights = next_cache.states[0]  # 如果使用cache.updates那么保留了所有chunk单独的更新梯度，本质上没有混合seq信息
        for k,v in weights.items():  # 混合batch的权重使得测试时使用统一的权值
            weights[k] = torch.mean(v, dim=0, keepdim=True)
        gs_retrived_1 = self.g2s_Memory.retrieve_memories(g_key_seq, weights)
        g_pred, s_pred = gs_retrived_1[..., :self.g_size], gs_retrived_1[..., self.g_size:]
        gs_key_seq_with_noise2 = torch.cat((g_seq + torch.randn_like(g_seq) * 0.1, stoch_seq), dim=-1)
        gs_retrived_2 = self.g2s_Memory.retrieve_memories(gs_key_seq_with_noise2, weights)
        g_corr, s_corr = gs_retrived_2[..., :self.g_size], gs_retrived_2[..., self.g_size:]

        g_corr_loss = F.mse_loss(g_corr, g_seq)
        s_pred_loss = F.mse_loss(s_pred, stoch_seq.detach()) + F.mse_loss(s_pred.detach(), stoch_seq)
        identity_loss = F.mse_loss(g_pred, g_seq) + F.mse_loss(s_corr, stoch_seq)
        loss_detail = {
            # 'g_corr_loss': g_corr_loss,
            "s_pred_loss": s_pred_loss,
            # "identity_loss": identity_loss,
        }
        if return_cache:
            return loss_detail, next_cache, s_pred
        else:
            return loss_detail, None, s_pred

class BiMemNet(nn.Module):
    def __init__(self, dim, g_size, stoch_size, chunk_size=64, segment_size=128):
        super(BiMemNet, self).__init__()
        self.g2s_Memory = NeuralMemory(
            dim = g_size,
            chunk_size=chunk_size,
            batch_size=segment_size,  # 是根据batch_size将一个完整的seq分割成每个chunk 128的小段，所以这里叫batch_size不合理，
            qkv_receives_diff_views=True,
            max_grad_norm=10.0,
        )
        self.s2g_Memory = NeuralMemory(
            dim=stoch_size,
            chunk_size=chunk_size,
            batch_size=segment_size, qkv_receives_diff_views=True,
            max_grad_norm=10.0,
        )
        self.g_size = g_size
        self.stoch_size = stoch_size
        self.chunk_size = chunk_size
        self.segment_size = segment_size

    def get_weight_from_cache(self, cache, eval=True):
        # weights = cache.updates  # 训练过程中updates保留了过去所有步长的权重更新，和seq长度一致,使用该权重时不同seq使用不同权重
        weights = cache.weights  # states是按照seq维度衰减累加后的updates

        if eval:
            eval_weights = {}
            for k, v in weights.items():  # 混合batch的权重使得测试时使用统一的权值
                eval_weights[k] = torch.mean(v, dim=0, keepdim=True)
            return eval_weights
        return weights

    def step(self, g_seq, stoch_seq, cache=None, return_cache=False):
        kv_4g2s = torch.stack((g_seq, g_seq, stoch_seq))
        kv_4s2g = torch.stack((stoch_seq, stoch_seq, g_seq))
        if cache is not None:
            g2s_cahce, s2g_cahce = cache
        else:
            g2s_cahce, s2g_cahce = None, None
        pred_stoch, next_g2s_cahce = self.g2s_Memory.forward(
            kv_4g2s,
            state=g2s_cahce,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        g_corr, next_s2g_cache, g_surprise = self.s2g_Memory.forward(
            kv_4s2g,
            state=s2g_cahce,
            prev_weights=None,
            return_surprises=True
        )
        g_surprise = g_surprise[0].squeeze(1).unsqueeze(-1)  # B,H,S ->B,S,1其中H代表多头的头数量
        # weights = self.get_weight_from_cache(next_g2s_cahce, eval=True)
        # pred_stoch_by_corr = self.g2s_Memory.retrieve_memories(g_corr, weights=weights)
        next_cache = (next_g2s_cahce, next_s2g_cache)
        return pred_stoch, g_corr, g_surprise, next_cache

    def retrieve_memories(self, keys, cache, mem_name='g2s'):
        weights = self.get_weight_from_cache(cache, eval=True)
        if mem_name == 's2g':
            value = self.s2g_Memory.retrieve_memories(keys, weights)
        elif mem_name == 'g2s':
            value = self.g2s_Memory.retrieve_memories(keys, weights)
        else:
            raise NotImplementedError
        return value

    def compute_loss(self, g_seq, stoch_seq, cache=None, return_cache=False):
        kv_4g2s = torch.stack((g_seq, g_seq, stoch_seq))
        kv_4s2g = torch.stack((stoch_seq, stoch_seq, g_seq))
        if cache is not None:
            g2s_cahce, s2g_cahce = cache
        else:
            g2s_cahce, s2g_cahce = None, None
        pred_stoch, next_g2s_cahce = self.g2s_Memory.forward(
            kv_4g2s,
            state=g2s_cahce,
            prev_weights=None,  # 仅与layer有关，用于跨层残差更新
        )
        pred_g, next_s2g_cache = self.s2g_Memory.forward(
            kv_4s2g,
            state=s2g_cahce,
            prev_weights=None,
        )

        g2s_rep_loss = F.mse_loss(pred_stoch, stoch_seq.detach())
        g2s_dyn_loss = F.mse_loss(pred_stoch.detach(), stoch_seq)

        s2g_rep_loss = F.mse_loss(pred_g, g_seq.detach())
        s2g_dyn_loss = F.mse_loss(pred_g.detach(), g_seq)

        loss_detail = {
            'g2s': g2s_dyn_loss+g2s_rep_loss, 's2g': s2g_dyn_loss+s2g_rep_loss
        }  # 这里的损失只做可视化，不可用于训练，会导致
        # loss = (1.0*rep_loss + 1.0*dyn_loss).mean()
        next_cache = (next_g2s_cahce, None)
        if return_cache:
            return loss_detail, next_cache, pred_stoch
        return loss_detail, None, pred_stoch
