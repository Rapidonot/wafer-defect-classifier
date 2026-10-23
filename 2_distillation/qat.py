import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, scores
from student import Student, dihedral_t, kd_loss, predict

M_BITS = 15


# rounding that still lets gradients through during training
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


# round weights to 8-bit integers (one scale per output channel)
def fq_weight(w):
    s = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8) / 127
    return torch.clamp(RoundSTE.apply(w / s), -127, 127) * s, s.flatten()


# merge batch norm into the conv weights and bias, so the chip only does conv + bias
def fold_bn(conv, bn):
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * k.view(-1, 1, 1, 1), bn.bias - bn.running_mean * k


# the student with every weight and activation rounded to 8 bits, so it can be fine-tuned for 8-bit maths
class QStudent(nn.Module):

    def __init__(self, student: Student, act_scales):
        super().__init__()
        self.in_scale = student.input_scale
        convs = [m for m in student.features if isinstance(m, nn.Conv2d)]
        bns = [m for m in student.features if isinstance(m, nn.BatchNorm2d)]
        self.w = nn.ParameterList()
        self.b = nn.ParameterList()
        for c, bn in zip(convs, bns):
            w, b = fold_bn(c, bn)
            self.w.append(nn.Parameter(w.detach().clone()))
            self.b.append(nn.Parameter(b.detach().clone()))
        self.pools = [pool for _, pool in student.layers]
        self.hw = nn.Parameter(student.head.weight.detach().clone())
        self.hb = nn.Parameter(student.head.bias.detach().clone())
        self.register_buffer('act_scales', torch.tensor(act_scales, dtype=torch.float32))

    def forward(self, x):
        s_in = self.in_scale
        a = x.float()
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            wq, sw = fq_weight(w)
            bq = RoundSTE.apply(b / (s_in * sw)) * (s_in * sw)
            y = F.conv2d(a * s_in, wq, bq, padding=1)
            if self.pools[i]:
                y = F.max_pool2d(y, 2)
            s_out = self.act_scales[i]
            a = torch.clamp(RoundSTE.apply(y / s_out), 0, 255)  # activations become 0..255
            s_in = s_out
        g = a.mean((2, 3)) * s_in
        wq, sw = fq_weight(self.hw)
        bq = RoundSTE.apply(self.hb / (s_in * sw / a.shape[2] / a.shape[3])) * (s_in * sw / a.shape[2] / a.shape[3])
        return F.linear(g, wq, bq)


# measure how big each layer's outputs get, to choose its 8-bit scale
@torch.no_grad()
def calibrate(student, x, pct=99.99):
    student.eval()
    h = x.float() * student.input_scale
    scales = []
    for j, m in enumerate(student.features):
        h = m(h)
        if j % 4 == 3:
            scales.append(float(np.percentile(h.flatten().numpy(), pct)) / 255)
    return scales


# turn a real-valued rescale factor into an integer multiply M and a right shift S
def requant_params(real_mult):
    assert real_mult > 0
    s = 0
    while real_mult * 2 ** s < 2 ** (M_BITS - 1):
        s += 1
    m = int(round(real_mult * 2 ** s))
    if m == 2 ** M_BITS:
        m //= 2
        s -= 1
    return m, s


# write out the final integer weights, biases and rescale values
@torch.no_grad()
def export(q: QStudent, final_res):
    layers, s_in = [], q.in_scale
    for i, (w, b) in enumerate(zip(q.w, q.b)):
        _, sw = fq_weight(w)
        wi = torch.clamp(torch.round(w / sw.view(-1, 1, 1, 1)), -127, 127).to(torch.int8).numpy()
        bi = torch.round(b / (s_in * sw)).to(torch.int64).numpy()
        s_out = float(q.act_scales[i])
        mult = [requant_params(float(s_in * sw[o] / s_out)) for o in range(len(sw))]
        layers.append({'w': wi, 'b': bi.astype(np.int32), 'M': np.array([m for m, _ in mult], np.int32),
                       'S': np.array([s for _, s in mult], np.int32)})
        s_in = s_out
    _, sw = fq_weight(q.hw)
    hw = torch.clamp(torch.round(q.hw / sw.view(-1, 1)), -127, 127).to(torch.int8).numpy()
    hb = torch.round(q.hb / (s_in * sw / final_res ** 2)).to(torch.int64).numpy().astype(np.int32)
    return layers, {'w': hw, 'b': hb}


# 8-bit fine-tuning of a trained student, then export to results/<name>_int8.npz
def run_qat(name, data, teacher, epochs=10, lr=3e-4, alpha=0.3, temp=2.0):
    ck = torch.load(f'results/{name}.pt')
    st = Student(ck['chans'])
    st.input_scale = 1.0 / ck['factor'] ** 2
    st.load_state_dict(ck['state_dict'])
    xtr, ytr = data['train']
    yv = data['val'][1].numpy()
    float_val = scores(yv, predict(st, data['val'][0]))['macro_f1']

    dev = device()
    q = QStudent(st, calibrate(st, xtr[:4096])).to(dev)
    ptq_val = scores(yv, predict(q, data['val'][0]))['macro_f1']
    print(f'{name}: float val {float_val:.4f}  8-bit before fine-tuning {ptq_val:.4f}', flush=True)

    opt = torch.optim.Adam(q.parameters(), lr=lr)
    xd, yd, td = xtr.to(dev), ytr.to(dev), teacher['train'].to(dev)
    best = (ptq_val, {k: v.detach().cpu().clone() for k, v in q.state_dict().items()})
    for ep in range(epochs):
        q.train()
        perm = torch.randperm(len(xd), device=dev)
        for i in range(0, len(xd), 512):
            idx = perm[i:i + 512]
            k = int(torch.randint(8, (1,)))
            loss = kd_loss(q(dihedral_t(xd[idx], k)), yd[idx], td[idx, k], alpha, temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = scores(yv, predict(q, data['val'][0]))['macro_f1']
        print(f'{name}: 8-bit fine-tuning epoch {ep + 1:02d} val {v:.4f}', flush=True)
        if v > best[0]:
            best = (v, {k: t.detach().cpu().clone() for k, t in q.state_dict().items()})
    q.load_state_dict(best[1])
    q.cpu()

    final_res = data['val'][0].shape[-1] // 2 ** sum(q.pools)
    layers, head = export(q, final_res)
    np.savez(f'results/{name}_int8.npz', n_layers=len(layers), head_w=head['w'], head_b=head['b'],
             **{f'l{i}_{k}': v for i, L in enumerate(layers) for k, v in L.items()},
             **{f'l{i}_pool': np.int32(p) for i, p in enumerate(q.pools)})
    out = {'float_val': float_val, 'ptq_val': ptq_val, 'qat_val': best[0]}
    json.dump(out, open(f'results/{name}_qat.json', 'w'), indent=2)
    return out
