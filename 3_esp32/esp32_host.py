import argparse
import glob
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '2_distillation'))
import intref
from common import CLASSES, TEST_PATH, load_split, scores
from student import to_planes

MODEL = HERE.parent / '2_distillation/results/student_int8.npz'
REPLY = struct.Struct('<BB8iI')  # 0x5A, defect mask, 8 logits, time in microseconds


# open the board's USB serial port (DTR/RTS off, otherwise the ESP32-S3 resets into download mode)
def open_board(port):
    import serial
    ports = sorted(glob.glob('/dev/cu.usbmodem*'))
    if not port and not ports:
        sys.exit('No /dev/cu.usbmodem* port: is the ESP32-S3 plugged in?')
    port = port or ports[-1]
    link = serial.Serial()
    link.port, link.baudrate, link.timeout = port, 115200, 5
    link.dtr = link.rts = False
    link.open()
    time.sleep(2.0)  # opening the port restarts the board; wait for the sketch to start
    return link, port


# send one wafer and read back the board's answer
def classify(link, counts):
    link.reset_input_buffer()
    t0 = time.perf_counter()
    link.write(bytes([0xA5]) + counts.astype(np.uint8).tobytes())
    raw = link.read(REPLY.size)
    dt = time.perf_counter() - t0
    if len(raw) != REPLY.size or raw[0] != 0x5A:
        raise RuntimeError(f'bad or missing reply ({len(raw)} bytes)')
    _, mask, *rest = REPLY.unpack(raw)
    return mask, np.array(rest[:8], np.int64), rest[8], dt


# run wafers on the board and compare every output with the Python integer model
def run(link, layers, head, x, ids):
    counts = to_planes(x[ids])
    expected = intref.run(layers, head, counts)
    logits, us, rt, bad = np.zeros((len(ids), 8), np.int64), [], [], 0
    for i, c in enumerate(counts):
        _, lg, t_us, dt = classify(link, c)
        logits[i] = lg
        us.append(t_us)
        rt.append(dt)
        bad += not np.array_equal(lg, expected[i])
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(ids)} wafers, {bad} mismatches', flush=True)
    return logits, np.array(us) / 1e3, np.array(rt) * 1e3, bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['demo', 'evaluate'])
    p.add_argument('--idx', type=int, default=2119, help='demo: which test wafer')
    p.add_argument('--n', type=int, default=0, help='evaluate: number of test wafers (0 = all)')
    p.add_argument('--port')
    a = p.parse_args()
    layers, head = intref.load_npz(MODEL)
    x, y = load_split(TEST_PATH)
    ids = np.array([a.idx]) if a.cmd == 'demo' else np.arange(a.n or len(x))
    link, port = open_board(a.port)
    try:
        logits, ms, rt, bad = run(link, layers, head, x, ids)
    finally:
        link.close()

    if a.cmd == 'demo':
        truth = [CLASSES[k] for k in range(8) if y[a.idx][k]] or ['none']
        pred = [CLASSES[k] for k in range(8) if logits[0][k] > 0] or ['none']
        print(f'test wafer {a.idx}: truth {", ".join(truth)} | ESP32-S3 predicts {", ".join(pred)} | '
              f'matches Python model {bad == 0} | network {ms[0]:.1f} ms | round trip {rt[0]:.0f} ms')
        return 0

    sc = scores(y[ids], logits)
    print(f'{port}: {len(ids) - bad}/{len(ids)} wafers identical to the Python model; '
          f'network {ms.mean():.1f} ms per wafer; USB round trip {rt.mean():.0f} ms')
    print(f'test macro F1 {sc["macro_f1"]:.4f}, micro F1 {sc["micro_f1"]:.4f} (from the board\'s outputs)')
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    path = out / f'evaluate_{time.strftime("%Y%m%d_%H%M")}.json'
    path.write_text(json.dumps({'board': 'ESP32-S3', 'n_wafers': len(ids), 'bit_exact_mismatches': bad,
                                'ms_mean': ms.mean(), 'ms_max': ms.max(), 'round_trip_ms': rt.mean(), **sc,
                                'date': time.strftime('%Y-%m-%d %H:%M')}, indent=2))
    print(f'saved {path.relative_to(HERE.parent)}')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
