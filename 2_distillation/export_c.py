import argparse
from pathlib import Path

import numpy as np

from intref import load_npz

ROOT = Path(__file__).resolve().parent.parent


# format a list of numbers as a C array
def c_array(ctype, name, vals, per_line=24):
    vals = [int(v) for v in np.asarray(vals).ravel()]
    lines = [', '.join(str(v) for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f'static {ctype} {name}[{len(vals)}] = {{\n    ' + ',\n    '.join(lines) + '\n};\n'


# write the 8-bit model as model_data.h for the ESP32 sketch
def write_header(layers, head, out):
    c, h, w = 2, 56, 56
    parts, descs, bufmax = [], [], (c * (h + 2) * (w + 2))
    for i, L in enumerate(layers):
        o = L['w'].shape[0]
        assert L['w'].shape[1] == c and np.all(L['S'] >= 1) and np.all(L['S'] <= 62) and np.all(L['M'] > 0)
        parts += [c_array('int8_t', f'L{i}_W', L['w']), c_array('const int32_t', f'L{i}_B', L['b']),
                  c_array('const int32_t', f'L{i}_M', L['M']), c_array('const uint8_t', f'L{i}_S', L['S'])]
        descs.append(f'    {{{c}, {o}, {h}, {w}, {L["pool"]}, L{i}_W, L{i}_B, L{i}_M, L{i}_S}},')
        if L['pool']:
            h, w = h // 2, w // 2
        c = o
        bufmax = max(bufmax, c * (h + 2) * (w + 2))  # biggest feature map, sizes the buffers in cnn.c
    assert head['w'].shape == (8, c)
    parts += [c_array('int8_t', 'HEAD_W', head['w']), c_array('const int32_t', 'HEAD_B', head['b'])]
    text = f"""#pragma once
#include <stdint.h>
#include "cnn.h"

#define N_LAYERS {len(layers)}
#define IN_C 2
#define IN_H 56
#define IN_W 56
#define HEAD_C {c}
#define LAST_H {h}
#define LAST_W {w}
#define MAP_BUF_BYTES {bufmax}

""" + '\n'.join(parts) + '\nstatic const cnn_layer_t LAYERS[N_LAYERS] = {\n' + '\n'.join(descs) + '\n};\n'
    out.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='results/student_int8.npz')
    ap.add_argument('--out', default=str(ROOT / '3_esp32' / 'wafer_esp32' / 'model_data.h'))
    a = ap.parse_args()
    layers, head = load_npz(a.npz)
    write_header(layers, head, Path(a.out))
    print(f'wrote {a.out}: {len(layers)} layers')


if __name__ == '__main__':
    main()
