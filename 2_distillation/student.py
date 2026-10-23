import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CLASSES, device, scores


# shrink a 224x224 wafer map to two 56x56 count maps: dies on the wafer, and dies that failed
def to_planes(x_uint8, factor=4):
    n, h, w = x_uint8.shape
    planes = np.stack([x_uint8 > 0, x_uint8 == 254], 1).astype(np.uint8)
    return planes.reshape(n, 2, h // factor, factor, w // factor, factor).sum((3, 5), dtype=np.uint8)


# flip/rotate a batch of wafers (k = 0..7 covers every orientation)
def dihedral_t(x, k):
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x


# '8,16,32,32n,48n' -> channels per layer; 'n' means no max pooling after that layer
def parse_layers(spec):
    if isinstance(spec, (list, tuple)):
        return [tuple(t) if isinstance(t, (list, tuple)) else (int(t), True) for t in spec]
    return [(int(t.rstrip('n')), not t.endswith('n')) for t in spec.split(',')]


class Student(nn.Module):

    def __init__(self, layers, in_ch=2, num_classes=len(CLASSES)):
        super().__init__()
        self.layers = parse_layers(layers)
        mods, c = [], in_ch
        for o, pool in self.layers:
            mods += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
            mods += [nn.MaxPool2d(2)] if pool else [nn.Identity()]
            c = o
        self.features = nn.Sequential(*mods)
        self.head = nn.Linear(c, num_classes)
        self.input_scale = 1.0

    def forward(self, x):
        x = self.features(x.float() * self.input_scale)
        return self.head(x.mean((2, 3)))  # global average pool, then one output per defect class


# model size in bytes and multiply-adds per wafer
def model_size(layers, res, in_ch=2, num_classes=len(CLASSES)):
    weights, macs, c, r = 0, 0, in_ch, res
    for o, pool in parse_layers(layers):
        weights += c * o * 9 + 4 * o
        macs += r * r * c * o * 9
        r = r // 2 if pool else r
        c = o
    weights += c * num_classes + 4 * num_classes
    macs += c * num_classes
    return {'weight_bytes': weights, 'macs': macs}


# distillation loss: part true labels, part the teacher's softened predictions
def kd_loss(logits, y, t_logits, alpha, temp):
    hard = F.binary_cross_entropy_with_logits(logits, y)
    soft = F.binary_cross_entropy_with_logits(logits / temp, torch.sigmoid(t_logits / temp)) * temp ** 2
    return alpha * hard + (1 - alpha) * soft


@torch.no_grad()
def predict(model, x, batch=2048):
    dev = next(model.parameters()).device
    model.eval()
    return torch.cat([model(x[i:i + batch].to(dev)).float().cpu() for i in range(0, len(x), batch)]).numpy()


def train(args, data, teacher):
    torch.manual_seed(args.seed)
    dev = device()
    xtr, ytr = data['train']
    ttr = teacher['train']
    layers = parse_layers(args.chans)
    model = Student(layers).to(dev)
    model.input_scale = 1.0 / args.factor ** 2
    # keep a running average of the weights; it is smoother than the raw weights, so it is what gets evaluated
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(xtr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    xtr_d, ytr_d, ttr_d = xtr.to(dev), ytr.to(dev), ttr.to(dev)
    best, t0 = (-1.0, None), time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), args.batch):
            idx = perm[i:i + args.batch]
            k = int(torch.randint(8, (1,)))  # random orientation, matched with the teacher's output for it
            xb = dihedral_t(xtr_d[idx], k)
            loss = kd_loss(model(xb), ytr_d[idx], ttr_d[idx, k], args.alpha, args.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update_parameters(model)
        val = scores(data['val'][1].numpy(), predict(ema.module, data['val'][0]))
        if val['macro_f1'] > best[0]:
            best = (val['macro_f1'], {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()})
        if (ep + 1) % 20 == 0 or ep + 1 == args.epochs:
            print(f"{args.name} ep {ep + 1:03d} loss {loss.item():.4f} val macroF1 {val['macro_f1']:.4f} "
                  f"best {best[0]:.4f} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best[1])
    torch.save({'state_dict': best[1], 'chans': layers, 'factor': args.factor}, f'results/{args.name}.pt')
    summary = {'name': args.name, 'chans': args.chans, 'best_val_macro_f1': best[0],
               'size': model_size(layers, 224 // args.factor)}
    json.dump(summary, open(f'results/{args.name}.json', 'w'), indent=2)
    return model, summary


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--chans', default='8,16,32,32n,48n')
    p.add_argument('--factor', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.3, help='weight on hard labels; 1-alpha on teacher')
    p.add_argument('--temp', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ema', type=float, default=0.999)
    return p
