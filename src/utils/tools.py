import numpy as np
import torch


def np_to_torch(img, device='cuda'):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img.float().div(255).unsqueeze(0).to(device)
    return img

def torch_to_np(img):
    img = img.mul(255).byte().detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    return img