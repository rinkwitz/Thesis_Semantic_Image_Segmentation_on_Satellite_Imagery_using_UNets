import torch
import torch.nn as nn


def DICE_coefficient(X, Y, zero_division_safe=False):
    # gets input of shape: torch.Size([batch_size, num_classes, 260, 260]) torch.Size([batch_size, 260, 260])
    # works for segementation and 2 classes
    if zero_division_safe:
        return (2 * torch.sum(X[:, 1, :, :] * Y)) / (torch.sum(X[:, 1, :, :] ** 2) + torch.sum(Y ** 2) + 1e-6)
    else:
        return (2 * torch.sum(X[:, 1, :, :] * Y)) / (torch.sum(X[:, 1, :, :] ** 2) + torch.sum(Y ** 2))


def IOU_loss(X, Y, device='cpu', zero_division_safe=False):
    # gets input of shape: torch.Size([batch_size, num_classes, 260, 260]) torch.Size([batch_size, 260, 260])
    # works for segementation and 2 classes
    Y_T = torch.ones(Y.shape, dtype=torch.int64, device=device) - Y
    Y2 = torch.empty((Y.shape[0], 2, Y.shape[1], Y.shape[2]), dtype=torch.int64, device=device)
    Y2[:, 0, :, :] = Y_T
    Y2[:, 1, :, :] = Y
    iou = torch.zeros(1, device=device)
    for i in range(2):
        other_idx = (i - 1) ** 2
        intersect = torch.sum(X[:, i, :, :] * Y2[:, i, :, :])
        union = torch.sum(Y2[:, i, :, :]) + torch.sum(X[:, i, :, :] * Y2[:, other_idx, :, :])
        iou += intersect / (union + 1e-6) if zero_division_safe else intersect / union
    return iou


if __name__ == '__main__':
    device = 'cpu'
    out = torch.tensor([[[[.9, .1, .1], [.1, .9, .9], [.9, .1, .9]], [[.1, .9, .9], [.9, .1, .1], [.1, .9, .1]]]]).to(
        device)
    # Y = torch.tensor([[[0, 1, 1], [1, 0, 0], [0, 1, 0]]]).to(device)
    Y = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]).to(device)
    print(out.shape, out.dtype, out.device)
    print(Y.shape, Y.dtype, Y.device)
    print(IOU_loss(out, Y, device=device, zero_division_safe=True))
