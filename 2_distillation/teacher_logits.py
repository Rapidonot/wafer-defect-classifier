import torch

from common import MEAN, STD, device


# run the ResNet-18 teacher on every wafer in all 8 flips/rotations and keep its raw outputs
@torch.no_grad()
def teacher_views(model, x_uint8, views=range(8), batch=1024):
    dev = device()
    model.to(dev).eval()
    mean, std = MEAN.to(dev), STD.to(dev)
    out = []
    for i in range(0, len(x_uint8), batch):
        xb = torch.from_numpy(x_uint8[i:i + batch]).to(dev)
        per_view = []
        for k in views:
            v = torch.rot90(xb, k % 4, dims=(-2, -1))
            v = torch.flip(v, dims=(-1,)) if k >= 4 else v
            t = (v.float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1) - mean) / std
            per_view.append(model(t).float().cpu())
        out.append(torch.stack(per_view, 1))
    return torch.cat(out).numpy()
