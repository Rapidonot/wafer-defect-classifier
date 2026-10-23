import json
import os
import shutil
import time

import numpy as np
import torch

from common import CLASSES, RESNET_BCE_CKPT, TEST_PATH, TRAIN_PATH, VAL_PATH, load_split, predict_logits, \
    resnet18_bce, scores
from intref import run_intref
from qat import run_qat
from student import parser, to_planes, train
from teacher_logits import teacher_views

CHANS = '8,16,32,32n,48n'  # 5 conv layers; the last two keep the 7x7 map (no pooling)
SEEDS = (0, 1, 2)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# results/summary.md: the three runs, the saved model, and per-class F1 next to the teacher
def write_summary(report, path='results/summary.md'):
    runs, final, t = report['runs'], report['final'], report['teacher_test']
    test = np.array([r['int8_test'] for r in runs])
    size = runs[0]['size']
    lines = ['# Distilled student: results', '',
             f"Student: conv layers {CHANS} (n = no pooling), global average pooling, 8 outputs. "
             f"{size['weight_bytes']:,} bytes of int8 weights and int32 biases, {size['macs']:,} multiply-adds "
             f"per wafer.", '', f"Teacher ResNet-18 test macro F1: {t['macro_f1']:.4f}", '',
             '| Run | Seed | Float val | Int8 val | Int8 test |', '|---|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['name']}{' **(saved)**' if r['name'] == final['name'] else ''} | {r['seed']} | "
                     f"{r['float_val']:.4f} | {r['int8_val']:.4f} | {r['int8_test']:.4f} |")
    lines += ['', f"Int8 test macro F1 over the {len(runs)} runs: {test.mean():.4f} ± {test.std():.4f}", '',
              f"Saved model: `results/student_int8.npz` = {final['name']} (best validation score).", '',
              '| Class | ResNet-18 | Student (int8) |', '|---|---:|---:|']
    for c in CLASSES:
        lines.append(f"| {c} | {t['per_class_f1'][c]:.4f} | {final['int8_test_per_class'][c]:.4f} |")
    open(path, 'w').write('\n'.join(lines) + '\n')


def main():
    os.makedirs('results', exist_ok=True)
    torch.backends.cudnn.benchmark = True
    xtr, ytr = load_split(TRAIN_PATH)
    xv, yv = load_split(VAL_PATH)
    xt, yt = load_split(TEST_PATH)
    log(f'data: train {len(xtr)}, val {len(xv)}, test {len(xt)}')

    # 1. the teacher: the ResNet-18 from Part 1
    teacher_model = resnet18_bce()
    teacher_model.load_state_dict(torch.load(RESNET_BCE_CKPT, map_location='cpu'))
    report = {'teacher_test': scores(yt, predict_logits(teacher_model, xt))}
    log(f"teacher ResNet-18 test macroF1 {report['teacher_test']['macro_f1']:.4f}")

    # 2. the teacher's outputs for every training wafer in all 8 orientations
    tl = {'train': teacher_views(teacher_model, xtr), 'val': teacher_views(teacher_model, xv, [0])[:, 0],
          'test': teacher_views(teacher_model, xt, [0])[:, 0]}
    del teacher_model

    data = {'train': (torch.from_numpy(to_planes(xtr)), torch.from_numpy(ytr)),
            'val': (torch.from_numpy(to_planes(xv)), torch.from_numpy(yv)),
            'test': (torch.from_numpy(to_planes(xt)), torch.from_numpy(yt))}
    teacher = {k: torch.from_numpy(v.astype(np.float32)) for k, v in tl.items()}
    del xtr

    # 3. train the student three times, convert each run to 8 bits, and score it with the integer maths
    runs = []
    for seed in SEEDS:
        args = parser().parse_args(['--name', f'student_s{seed}', '--chans', CHANS, '--seed', str(seed)])
        _, s = train(args, data, teacher)
        q = run_qat(args.name, data, teacher)
        ir = run_intref(args.name, {'val': data['val'], 'test': data['test']})
        runs.append({'name': args.name, 'seed': seed, 'float_val': s['best_val_macro_f1'], 'size': s['size'],
                     'qat_val': q['qat_val'], 'int8_val': ir['val']['macro_f1'], 'int8_test': ir['test']['macro_f1'],
                     'int8_test_per_class': ir['test']['per_class_f1']})
        log(f"{args.name}: 8-bit val {runs[-1]['int8_val']:.4f} test {runs[-1]['int8_test']:.4f}")

    # 4. keep the run with the best validation score (the test set is only used for reporting)
    final = max(runs, key=lambda r: r['int8_val'])
    shutil.copyfile(f"results/{final['name']}_int8.npz", 'results/student_int8.npz')
    report.update({'runs': runs, 'final': final})
    json.dump(report, open('results/summary.json', 'w'), indent=2)
    write_summary(report)
    log(f"saved results/student_int8.npz ({final['name']}) and results/summary.md")


if __name__ == '__main__':
    main()
