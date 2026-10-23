import json

import numpy as np

from common import scores


# load an exported 8-bit model (.npz)
def load_npz(path):
    z = np.load(path)
    layers = [{k: z[f'l{i}_{k}'] for k in ('w', 'b', 'M', 'S')} for i in range(int(z['n_layers']))]
    for i, L in enumerate(layers):
        L['pool'] = int(z[f'l{i}_pool'])
    return layers, {'w': z['head_w'], 'b': z['head_b']}


# 3x3 convolution with zero padding, all in integers
def conv3x3_acc(a, w, b):
    n, c, h, wd = a.shape
    p = np.pad(a.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    acc = np.zeros((n, w.shape[0], h, wd), np.int64)
    w64 = w.astype(np.int64)
    for dy in range(3):
        for dx in range(3):
            acc += np.einsum('nchw,oc->nohw', p[:, :, dy:dy + h, dx:dx + wd], w64[:, :, dy, dx], optimize=True)
    return acc + b.astype(np.int64).reshape(1, -1, 1, 1)


# 2x2 max pooling (an odd last row/column is dropped)
def pool2_max(acc):
    n, o, h, w = acc.shape
    acc = acc[:, :, :h // 2 * 2, :w // 2 * 2]
    return acc.reshape(n, o, h // 2, 2, w // 2, 2).max((3, 5))


# scale the sums back down to 0..255: (acc * M + rounding) >> S, then clip (this clip is also the ReLU)
def requant(acc, m, s):
    m = m.astype(np.int64).reshape(1, -1, 1, 1)
    s = s.astype(np.int64).reshape(1, -1, 1, 1)
    return np.clip((acc * m + (np.int64(1) << (s - 1))) >> s, 0, 255)


# the whole 8-bit network in integer maths; the ESP32's C code computes exactly the same numbers
def run(layers, head, counts):
    a = counts.astype(np.int64)
    for L in layers:
        acc = conv3x3_acc(a, L['w'], L['b'])
        if L['pool']:
            acc = pool2_max(acc)
        a = requant(acc, L['M'], L['S'])
    g = a.sum((2, 3))
    return g @ head['w'].astype(np.int64).T + head['b'].astype(np.int64)


# score an exported model on the given splits using the integer maths
def run_intref(name, splits):
    layers, head = load_npz(f'results/{name}_int8.npz')
    out = {}
    for split, (c, y) in splits.items():
        c, y = np.asarray(c), np.asarray(y)
        logits = np.concatenate([run(layers, head, c[i:i + 1024]) for i in range(0, len(c), 1024)])
        out[split] = scores(y, logits)
        print(f"{name}: {split} 8-bit macroF1 {out[split]['macro_f1']:.4f}", flush=True)
    json.dump(out, open(f'results/{name}_intref.json', 'w'), indent=2)
    return out
