import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import unsqueeze

from hpc_memory_net import MemNet, BiMemNet
from titans_pytorch import RelationNeuralMemory as Memory
import torch.distributions as torchd
from envs.data_utils import gen_data
from envs.data_utils import *
import matplotlib.pyplot as plt
import seaborn as sns

from titan_params import default_params, get_scaling_parameters
import utils


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
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sense, sample=True):
        logits = self.enc(sense)
        return logits

    def compute_loss(self, sense, stoch_logits=None, return_dec=False, reduction='mean'):  # decoder的训练可以考虑CE损失
        if stoch_logits is None:
            stoch = self(sense)
        else:
            stoch = stoch_logits

        data_shape = sense.shape
        pred = self.dec(stoch)
        sense_label = torch.argmax(sense, dim=-1).reshape(-1)
        pred_flatten = pred.reshape(-1, sense.shape[-1])
        loss = F.cross_entropy(pred_flatten, sense_label, reduction=reduction)

        if reduction == "none":
            loss = loss.reshape((data_shape[:-1]))
        pred_label = torch.argmax(pred, dim=-1).reshape(-1)
        acc = (pred_label==sense_label).float().mean()
        # loss = F.mse_loss(sense.detach(), pred)
        if return_dec:
            return loss, pred, acc
        return loss, acc

class GridModule(nn.Module):
    def __init__(self, action_dim, g_size, expand_factor=3, device='cuda:0'):
        super(GridModule, self).__init__()

        self.a2trans_layer = nn.Sequential(
            nn.Linear(action_dim, g_size*expand_factor,bias=False),
            nn.Tanh(),
            nn.Linear(g_size*expand_factor, g_size**2, bias=False),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def norm_g(self,g):
        # self.norm_g = nn.LayerNorm(g_size, eps=1e-8, elementwise_affine=False)
        clip_g = torch.clamp(g, min=-10.0, max=10.0)
        return clip_g
    

    def forward(self, prev_g, prev_a):
        g_shape = prev_g.shape
        W_trans = self.a2trans_layer(prev_a)
        W_trans = W_trans.reshape((*g_shape[:-1], self.g_size, self.g_size))
        g_delta = torch.bmm(W_trans, prev_g.unsqueeze(-1)).squeeze(-1)  # 效率上会优一些
        new_g = prev_g + g_delta
        new_g = self.norm_g(new_g)
        return new_g


class Dynamic(nn.Module):
    def __init__(self, para):
        super(Dynamic, self).__init__()
        self.par = para
        self.scaling=None
        self.grid = GridModule(self.par.n_actions, self.par.g_size, 
                                      device=self.par.device)
        self.visual_encdec = SimpleEncDec(self.par.s_size, self.par.s_size_project)
        # relational memory 
        self.memory = Memory( dim = self.par.mem_dim,
                              chunk_size = self.par.chunk_size,
                              batch_size = self.par.mem_batch_size,
                              heads = self.par.heads,
                              momentum_order =  self.par.momentum_order,
                              per_head_learned_parameters=self.par.heads>1,
                              # dim_head = self.par.dim_head,
                              learned_momentum_combine = self.par.learned_momentum_combine )
                 
        # project g  TODO adding layernorm layer
        self.projection_g = nn.Linear(self.par.g_size, self.par.g_size_project, bias=False)
        nn.init.orthogonal_(self.projection_g.weight)
        
        # correct g (p2g_mu)
        self.p2g_mu = nn.Sequential(
            nn.Linear(self.par.g_size_project, 2 * self.par.g_size),
            nn.ELU(),
            nn.Linear(2 * self.par.g_size, self.par.g_size)
        )
        for m in self.p2g_mu.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # p2g_var
        self.p2g_delta = nn.Sequential(
             nn.Linear(2 * self.par.g_size+1, 10 * self.par.g_size),
             nn.ELU(),
             nn.Linear(10 * self.par.g_size, self.par.g_size),
             nn.Sigmoid()
            )
        for m in self.p2g_delta.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)        

        # predict s (MLP_x_pred)
        self.MLP_x_pred = nn.Sequential(
            nn.Linear(self.par.s_size_project, self.par.s_size_hidden),
            nn.ELU(),
            nn.Linear(self.par.s_size_hidden, self.par.s_size_project)
        )
        for m in self.MLP_x_pred.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def step(self, a_i, x_proj, last_g, cache, t=0):
        # cache coxiaolong zou neuroscience_gen
        g_gen = self.grid(last_g, a_i)
        g_proj = self.projection_g(g_gen)
        # x_proj = s_i

        # memory 
        if t >0:
            # pred from g
            weight = cache.weights
            mem_in = g_proj.unsqueeze(1)
            # mem_in = torch.concat([g_proj, torch.zeros_like(x_proj)],dim=-1).unsqueeze(dim=1) #(B,1,D)
            gx_gen = self.memory.retrieve_memories(mem_in, weights=cache.weights).squeeze(1)
            # gx = self.memory(seq=mem_in, store_seq=None, state = cache, detach_mem_state=True)
            # g_gen_retr, x_gen_retr = gx.split([self.par.g_size_project, self.par.s_size_project],dim=-1)
            x_gen_pred = self.MLP_x_pred(gx_gen)

            # integrate, given g+x
            mem_in = (g_proj+x_proj).unsqueeze(1)
            # mem_in = torch.concat([g_proj, x_proj],dim=-1).unsqueeze(dim=1) #(B,1,D)
            gx_retr = self.memory.retrieve_memories(mem_in, weights=cache.weights).squeeze(1)
            # gx = self.memory(seq=mem_in, store_seq=None, state = cache, detach_mem_state=True)
            # g_retr, x_retr = gx.split([self.par.g_size_project, self.par.s_size_project], dim=-1)
            # x_retr, g_retr = gx.split([self.par.s_size_project, self.par.g_size_project],dim=-1)
            # g_retr, x_retr = gx, gx
            x_retr_pred = self.MLP_x_pred(gx_retr)

            # correct, g_corr = g_old + scalling*delta*(g-g_old)
            g_mu = self.p2g_mu(gx_retr)
            error_signal = F.mse_loss(x_retr_pred, x_proj.reshape(x_retr_pred.shape), reduction="none")
            error_signal = error_signal.mean(dim=-1,keepdims=True).detach()
            delta_in = torch.concat([g_mu, g_gen, error_signal],dim=-1)
            g_delta = self.p2g_delta(delta_in)
            g_int = g_gen + (g_mu- g_gen) * self.scaling.p2g_scale * g_delta
            g_int_proj = self.projection_g(g_int)
            # new pred from g_corr
            mem_in = (g_int_proj+x_proj).unsqueeze(1)
            # mem_in = torch.concat([g_int, x_proj], dim=-1).unsqueeze(dim=1)
            gx_int = self.memory.retrieve_memories(mem_in, weights=cache.weights).squeeze(1)
            # gx = self.memory(seq=mem_in, store_seq=None, state = cache, detach_mem_state=True)
            # _, x_int = gx.split([self.par.g_size_project, self.par.s_size_project], dim=-1)
            # x_int, _ = gx.split([self.par.s_size_project, self.par.g_size_project],dim=-1)
            x_int_pred = self.MLP_x_pred(gx_int)
        else:  # t=0时记忆没有内容，只能使用现有内容
            g_int = g_gen
            g_int_proj = g_proj
            x_gen_pred = x_proj
            x_int_pred = x_proj

        # storage
        mem_in = (g_int_proj+x_proj).unsqueeze(1)
        # mem_in = torch.concat([g_int, x_proj],dim=-1).unsqueeze(dim=1) #(B,1,D)
        cache = self.memory(seq=mem_in, state = cache, detach_mem_state=False)

        return (g_gen, g_int, x_gen_pred, x_int_pred, cache)
        

    def forward(self, act_seq, sen_seq, last_g = None, cache=None):
        self.train()
        B, T, _ = act_seq.shape # BxTxD

        x_seq = self.visual_encdec(sen_seq)
        if last_g is None:
            last_g = self.grid.g0.repeat(B,1)
        
        g_gen_arr = []
        g_int_arr = []
        x_gen_arr = []
        x_int_arr = []
        
        for t in range(T):
            g_gen, last_g, x_gen_pred, x_int_pred, cache = self.step(act_seq[:, t], x_seq[:, t], last_g, cache, t)
            
            g_gen_arr.append(g_gen)
            g_int_arr.append(last_g)
            x_gen_arr.append(x_gen_pred)
            x_int_arr.append(x_int_pred)
            
        res_dict = {}
        res_dict["g_gen"] = torch.stack(g_gen_arr,dim=1) # (B,T,D)
        res_dict["g_int"] = torch.stack(g_int_arr,dim=1) # (B,T,D)
        res_dict["x_gen"] = torch.stack(x_gen_arr,dim=1) # (B,T,D)
        res_dict["x_int"] = torch.stack(x_int_arr,dim=1) # (B,T,D)

        return res_dict 


    def compute_loss(self,res, sen_seq, s_visited):

        lx_gen = 0.0 
        lx_gint = 0.0 
        lg = 0.0
        eps= 1e-8
        norm, s_vis = 1.0, 1.0
        
        labels = sen_seq.argmax(dim=-1) #(B,T,D)
        # encdec_loss, _ = self.visual_encdec.compute_loss(sen_seq)

        lx_gen_, _ = self.visual_encdec.compute_loss(sen_seq, res["x_gen"], reduction="none")
        lx_gint_, _ = self.visual_encdec.compute_loss(sen_seq, res["x_int"], reduction="none")
        lg_ = F.mse_loss(res["g_gen"], res["g_int"], reduction="none").mean(-1)

        if self.par.train_on_visited_states_only:
            s_vis = s_visited
            batch_vis = torch.sum(s_vis) + eps
            # normalize sequence length
            norm = 1.0 / batch_vis
        lx_gen = (lx_gen_ * s_vis).sum()*norm * self.par.lx_gt_val
        lx_gint = (lx_gint_ * s_vis).sum() * norm
        lg = (lg_ * s_vis).sum() * self.par.lg_val * norm

        # losses
        # for t in range(self.par.seq_len):
        #     lx_gen_ = F.cross_entropy(res["x_gen"][t], labels[:,t], reduction="none")  # 应该是g2s->labels吧？
        #     lx_gint_ = F.cross_entropy(res["x_int"][t], labels[:,t], reduction="none")
        #
        #     lg_ = F.mse_loss(res["g_gen"][t], res["g_int"][t], reduction="none").mean(-1)
        #
        #     if self.par.train_on_visited_states_only:
        #         s_vis = s_visited[:, t]
        #         batch_vis = torch.sum(s_vis) + eps
        #         # normalize sequence length
        #         norm = 1.0 / (batch_vis * self.par.seq_len)
        #
        #     lx_gen += torch.sum(lx_gen_ * s_vis) * norm * self.par.lx_gt_val
        #     lx_gint += torch.sum(lx_gint_ * s_vis) * norm
        #     # pdb.set_trace()
        #     lg += torch.sum(lg_ * s_vis) * self.par.lg_val * norm
        
        losses = utils.DotDict()
        cost_all = 0.0 

        losses.lx_gen = lx_gen
        losses.lx_gint = lx_gint
        # losses.encdec = encdec_loss

        if 'lx_gen' in self.par.which_costs:
            cost_all += lx_gen * (1 + self.scaling.g_gen)
        if 'lx_gint' in self.par.which_costs:
            cost_all += lx_gint * (1 - self.scaling.g_gen)
        if 'lg' in self.par.which_costs:
            cost_all += lg * self.scaling.temp * self.par.lg_temp
            losses.lg = lg * self.scaling.temp * self.par.lg_temp
            losses.lg_unscaled = lg

        # cost_all += encdec_loss * self.par.l_encdec
        
        losses.train_loss = cost_all
        return losses


class Trainer:
    def __init__(self):
        self.par = default_params()
        self.device = self.par.device
        self.fix_train_env = init_env(self.par)
        self.test_env = init_env(self.par)
        self.dynamic = Dynamic(self.par)
        self.opt = torch.optim.Adam(self.dynamic.parameters(), lr=1e-3)
        # self.schedule = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[1000, 2000, 6000], gamma=0.2)

    def seq_train(self, num_episode=200):
        self.dynamic.to(self.device)
        self.dynamic.train()
        for e in range(num_episode):
            lr_scaling = get_scaling_parameters(e, self.par)
            self.dynamic.scaling = lr_scaling
            env = init_env(self.par)
            data = gen_data(env, self.par)

            act = torch.from_numpy(data["dirs"]).to(self.device).float()
            sen_seq = torch.from_numpy(data["sense"]).to(self.device).float() # (B, T, D)
            s_visited = torch.from_numpy(data["visit"]).to(self.device).float() # (B,T)
            
            res = self.dynamic(act, sen_seq, last_g = None, cache=None)
            loss = self.dynamic.compute_loss(res, sen_seq, s_visited)
            
            self.opt.zero_grad()
            loss.train_loss.backward()
            self.opt.step()

            if e % 20 == 0:
                print(f"{e}: x_gen={loss.lx_gen.item():.2f}, "
                      f"x_int={loss.lx_gint.item():.2f}, g_corr={loss.lg_unscaled.item():.2f}")
                torch.save(self.dynamic.state_dict(), "../saved_models/gms_net.pth")

    @torch.no_grad()
    def eval(self, load=True):
        if load:
            ckpt = torch.load("../saved_models/gms_net.pth")
            self.dynamic.load_state_dict(ckpt)
            self.dynamic.to(self.device)

            lr_scaling = get_scaling_parameters(100, self.par)
            self.dynamic.scaling = lr_scaling
        env = init_env(self.par)
        data = gen_data(env, self.par)

        act = torch.from_numpy(data["dirs"]).to(self.device).float()
        sen_seq = torch.from_numpy(data["sense"]).to(self.device).float()  # (B, T, D)
        s_visited = torch.from_numpy(data["visit"]).to(self.device).float()  # (B,T)

        res = self.dynamic(act, sen_seq, last_g=None, cache=None)
        g_proj = self.dynamic.projection_g(res["g_int"])
        g_gen_proj = self.dynamic.projection_g(res["g_gen"])
        s_proj = self.dynamic.visual_encdec(sen_seq)

        mem_in = g_proj + s_proj
        print(mem_in.shape)
        cache = None
        for i in range(10):
            cache = self.dynamic.memory(seq=mem_in, state=cache, detach_mem_state=False)

            gx_full = self.dynamic.memory.retrieve_memories(mem_in, weights=cache.weights)
            gx_only_g = self.dynamic.memory.retrieve_memories(g_proj, weights=cache.weights)
            gx_gen = self.dynamic.memory.retrieve_memories(g_gen_proj, weights=cache.weights)

            remem_full_err = F.mse_loss(gx_full, mem_in)
            remem_from_g = F.mse_loss(gx_only_g, mem_in)
            remem_from_gen = F.mse_loss(gx_gen, mem_in)
            print("err by full", remem_full_err.item(), remem_from_g.item(), remem_from_gen.item())

        pdb.set_trace()

    def pretrain(self):
        self.dynamic.to(self.device)
        self.dynamic.train()
        for e in range(10000):
            lr_scaling = get_scaling_parameters(e, self.par)
            self.dynamic.scaling = lr_scaling
            env = init_env(self.par)
            data = gen_data(env, self.par)

            act = torch.from_numpy(data["dirs"]).to(self.device).float()
            sen_seq = torch.from_numpy(data["sense"]).to(self.device).float()  # (B, T, D)
            with torch.no_grad():
                res = self.dynamic(act, sen_seq, last_g=None, cache=None)
                g_proj = self.dynamic.projection_g(res["g_int"])
                g_gen_proj = self.dynamic.projection_g(res["g_gen"])
                s_proj = self.dynamic.visual_encdec(sen_seq)

            mem_in = g_proj + s_proj
            cache = None
            # 记忆的三个损失
            cache = self.dynamic.memory(seq=mem_in, state=cache, detach_mem_state=False)
            gx_full = self.dynamic.memory.retrieve_memories(mem_in, weights=cache.weights)
            # print("re 1 gpu use", torch.cuda.memory_allocated() // 1024 ** 2)
            # for t in range(3):
            #     gx_only_g = self.dynamic.memory.retrieve_memories(g_proj, weights=cache.weights)
            #     g_proj = gx_only_g
            # # print("re 2gpu use", torch.cuda.memory_allocated() // 1024 ** 2)
            # gx_gen = self.dynamic.memory.retrieve_memories(g_gen_proj, weights=cache.weights)

            remem_full_err = F.mse_loss(gx_full, mem_in)
            # remem_from_g = F.mse_loss(gx_only_g, mem_in)
            # remem_from_gen = F.mse_loss(gx_gen, mem_in)

            loss = remem_full_err # + remem_from_g + remem_from_gen

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if e % 20 == 0:
                print("err by full", remem_full_err.item(),)# remem_from_g.item(), remem_from_gen.item())
                torch.save(self.dynamic.state_dict(), "../saved_models/pretrain_memory.pth")


    # def eval(self, load=True):
    #     if load:
    #         ckpt = torch.load("./saved_models/dynamic_store_once_and_retrive.pth")
    #         self.dynamic.load_state_dict(ckpt)
    #         self.dynamic.to(self.device)

    #     traj_data = gen_data(self.fix_train_env, self.par)
    #     pos = torch.from_numpy(traj_data["pos"]).to(self.device).long()
    #     act = torch.from_numpy(traj_data["dirs"]).to(self.device).float()  # a0是固定值，代表着到p0的动作
    #     sense = torch.from_numpy(traj_data["sense"]).to(self.device).float()
    #     with torch.no_grad():
    #         self.dynamic.evaluate(pos, sense)

    # @torch.no_grad()
    # def visual(self):
    #     ckpt = torch.load("./saved_models/dynamic.pth")
    #     self.dynamic.load_state_dict(ckpt)
    #     self.dynamic.to(self.device)
    #     pos = torch.arange(0, self.fix_train_env.n_states).to(self.device).long()
    #     sense = sample_data(pos, self.fix_train_env.states_mat, s_size=self.par.s_size)
    #     raise NotImplementedError
    #     pos_emb = self.dynamic.grid_module.gen_seq_g(pos).cpu().numpy()
    #     # visual_emb = self.dynamic.encdec(sense)

    #     plt.figure(figsize=(10,6))
    #     sns.heatmap(pos_emb, cmap='coolwarm', xticklabels=4)
    #     plt.show()


if __name__ == '__main__':
    seed = 101
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    trainer = Trainer()
    # trainer.seq_train(num_episode=10000)
    # trainer.visual()
    # trainer.eval()
    trainer.pretrain()