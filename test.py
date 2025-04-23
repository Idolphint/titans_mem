import pdb
from tensordict import TensorDict
import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from titans_pytorch.memory_models import MemoryMLP, ResidualNorm
from titans_pytorch.neural_memory import *
from torch.func import functional_call, vmap, grad


def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class MyNeuralMemory(nn.Module):
    def __init__(self, dim=64):
        super(MyNeuralMemory, self).__init__()
        self.dim = dim
        # s1. 定义记忆模型与可训练参数
        model = MemoryMLP(
                    dim=dim,
                    depth=3,
                    expansion_factor=4,
                ).cuda()
        model = ResidualNorm(dim=dim, model=model)
        self.memory_model = model

        mem_model_params = dict(model.named_parameters())
        self.num_memory_parameter_tensors = len(mem_model_params)
        self.memory_model_parameter_names = [*mem_model_params.keys()]
        memory_model_parameters = [*mem_model_params.values()]
        self.memory_model_parameters = ParameterList(memory_model_parameters)

        # s2. 定义参数更新计算函数
        def forward_and_loss(params, inputs, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            # if loss_weights is not None:
            #     weighted_loss = loss * loss_weights
            # else:
            weighted_loss = loss
            return weighted_loss.sum(), loss

        grad_fn = grad(forward_and_loss, has_aux=True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims=(0, 0, 0))
        self.store_memory_loss_fn = default_loss_fn

        # 其他辅助层
        self.assoc_scan = AssocScan(use_accelerated=False)
        self.to_decay_factor = Sequential(  # B,S,dim - > B,S,1
            nn.Linear(dim, 1),
        )

        self.store_norm = nn.RMSNorm(dim)
        self.max_grad_norm = 5.0

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(self, batch):  # TODO 允许不同的batch拥有不同的mem？
        weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> b ...', b = batch)
        return weights

    def store_memories(self, seq, weights,
                       past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,  # 用于动量更新的之前的记忆状态
                       seq_index=0,
                       return_surprises=True,
                    ):
        _, batch, seq_len = seq.shape[:3]
        next_seq_len_index = seq_index + seq_len  # 这里没有用chunk避免引入错误
        remainder = None

        if weights is None:  # 这个weight只适用于过去的累计权重，所以没有seq维度，已经把所有时间都压缩到一起了
            weights = self.init_weights(batch)
            print("init weight")
        print("check ori weights.0 2batch", weights["model.weights.0"][:, 0, :10])
        # 用于更新的权重存在seq维度
        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n=seq_len)
        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')
        # print(batch, seq_len)
        # for para_name, wei in weights_for_surprise.items():
        #     print(para_name, wei.shape,weights[para_name].shape)
        seq = self.store_norm(seq)

        key_seq, value_seq = seq
        # decay_factor = self.to_decay_factor(key_seq).sigmoid()
        decay_factor = torch.ones((batch, seq_len, 1)).cuda() * 0.9999

        keys, values = (rearrange(t, 'b n ... -> (b n) 1 ...') for t in (key_seq, value_seq))

        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, values)
        grads = TensorDict(grads)
        grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))
        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b=batch)
        surprises = grads.mul(-1)
        print("check grads 2B2S", grads["model.weights.0"][:, :2, 0, :10])

        if not exists(past_state):  # surprise的计算需要过去seq的信息
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper
            minibatch_init_weight = weights
            init_momentum = None  # self.init_momentum(batch)
            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # 这里会根据每一层逐步更新
        updates = TensorDict()
        next_last_update = TensorDict()
        next_last_momentum = None
        # print("decay", decay_factor.mean(), decay_factor.shape)
        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):
            update = surprise
            # TODO 增加动量系统
            print(param_name, surprise.mean().item())  # decay_factor能否常数化？
            # print(decay_factor.shape, update.shape, last_update.shape, param_name)
            # pdb.set_trace()
            # update = last_update + (decay_factor * update).mean(dim=1)  # 模拟梯度下降
            update = self.assoc_scan(1. - decay_factor, update, prev=last_update, remove_prev=False)
            # print("after update", update.shape, update[0,0,0], param_name)
            if param_name == "model.weights.0":
                print("last update", last_update[0, 0, :3], "new update", update[0,:2, 0, :3])
            # next_last_update[param_name] = update
            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]  # 选择逐seq更新后的最后一次的weights作为最终权重
        next_state = (next_last_update, next_last_momentum)
        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)
        pdb.set_trace()
        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, None)

    def forward(
            self, seq,
            state: NeuralMemState | None = None,  # 存储着seq_index之前的模型权重，惊喜更新等等
            detach_mem_state = False, return_surprises=False
        ):
        retrived_seq, seq = seq[0], seq[1:]  # q, [k-v]
        if not exists(state):
            state = (0, None, None, None, None)
        seq_index, weights, cache_store_seq, past_state, updates = state
        # 由于不分chunk所以不存在remainder

        def accum_updates(past_updates, future_updates):
            if not exists(past_updates):
                return future_updates
            return TensorDict({param_name: cat((past_update[:, :-1], future_update), dim = 1) for (param_name, past_update), (_, future_update) in zip(past_updates.items(), future_updates.items())})

        # 存入内存
        next_updates, next_neural_mem_state, surprises = self.store_memories(
            seq, weights,past_state=past_state, seq_index=seq_index,return_surprises=True)
        weights = next_neural_mem_state.weights
        seq_index = next_neural_mem_state.seq_index
        past_state = next_neural_mem_state.states

        # 更新权重和保存量
        updates = accum_updates(updates, next_updates)
        last_update, last_momentum = past_state
        weights = last_update

        next_neural_mem_state = next_neural_mem_state._replace(
            weights=weights,  # TODO 需要注意weights是否被替换？
            states=past_state,
            updates=updates,
        )

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning
        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises

# TODO 通过不断存入内存，检查suprise是否降低


mm_model = MyNeuralMemory().cuda()

g_store = torch.sign(torch.randn(1024, 64)).cuda()  # 假设存在一个1024的环境，存储了不同的g-s
s_store = torch.randn(1024, 64).cuda()
# ind = torch.randperm(1024).cuda()[:222]
# seq = torch.randn(2, 1024, 384).cuda()  #B, S, D
cache, weights, past_state = None, None, None
for e in range(1000):
    ind = torch.randperm(1024).cuda()[:222]
    g = g_store[ind].reshape(2, 111, -1)
    s = s_store[ind].reshape(2, 111, -1)
    # print(s[0,0], g[0,0])
    seq = torch.stack((g, s), dim = 0)

    next_updates, next_neural_mem_state, surprises = mm_model.store_memories(
        seq, weights, past_state=past_state, seq_index=0, return_surprises=True)
    # weights = next_neural_mem_state.weights
    seq_index = next_neural_mem_state.seq_index
    past_state = next_neural_mem_state.states
    last_update, last_momentum = past_state
    weights = last_update
    print("surprise=", surprises[0].mean())
    # pdb.set_trace()
    # retrieved, mem_state = mem(seq)
# pdb.set_trace()