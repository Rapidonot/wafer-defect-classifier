# Distilled student: results

Student: conv layers 8,16,32,32n,48n (n = no pooling), global average pooling, 8 outputs. 29,904 bytes of int8 weights and int32 biases, 3,387,264 multiply-adds per wafer.

Teacher ResNet-18 test macro F1: 0.9657

| Run | Seed | Float val | Int8 val | Int8 test |
|---|---:|---:|---:|---:|
| student_s0 | 0 | 0.9446 | 0.9482 | 0.9433 |
| student_s1 | 1 | 0.9464 | 0.9480 | 0.9510 |
| student_s2 **(saved)** | 2 | 0.9461 | 0.9489 | 0.9406 |

Int8 test macro F1 over the 3 runs: 0.9450 ± 0.0044

Saved model: `results/student_int8.npz` = student_s2 (best validation score).

| Class | ResNet-18 | Student (int8) |
|---|---:|---:|
| Center | 0.9848 | 0.9842 |
| Donut | 0.9973 | 0.9960 |
| Edge-Loc | 0.9589 | 0.9421 |
| Edge-Ring | 0.9878 | 0.9848 |
| Loc | 0.9720 | 0.9612 |
| Near-full | 0.9114 | 0.8493 |
| Random | 0.9891 | 0.9759 |
| Scratch | 0.9246 | 0.8313 |
