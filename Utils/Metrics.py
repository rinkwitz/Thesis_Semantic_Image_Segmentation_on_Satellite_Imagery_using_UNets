import torch


def calc_Accuracy(pred, y):
    num_total = 1
    for dim in y.shape:
        num_total *= dim
    return (torch.sum(torch.eq(pred, y)) / num_total).item()


def calc_IOU(pred, y, zero_division_safe=False):
    if zero_division_safe:
        return (torch.sum(torch.logical_and(pred, y)) / (torch.sum(torch.logical_or(pred, y)) + 1e-8)).item()
    else:
        return (torch.sum(torch.logical_and(pred, y)) / torch.sum(torch.logical_or(pred, y))).item()


def calc_F1(pred, y, zero_division_safe=False):
    precision = calc_Precision(pred, y, zero_division_safe)
    recall = calc_Recall(pred, y, zero_division_safe)
    if zero_division_safe:
        return (2 * precision * recall) / (precision + recall + 1e-8)
    else:
        return (2 * precision * recall) / (precision + recall)


def calc_Precision(pred, y, zero_division_safe=False):
    if zero_division_safe:
        return (torch.sum(torch.logical_and(pred, y)) / (torch.sum(pred) + 1e-8)).item()
    else:
        return (torch.sum(torch.logical_and(pred, y)) / torch.sum(pred)).item()


def calc_Recall(pred, y, zero_division_safe=False):
    if zero_division_safe:
        return (torch.sum(torch.logical_and(pred, y)) / (torch.sum(y) + 1e-8)).item()
    else:
        return (torch.sum(torch.logical_and(pred, y)) / torch.sum(y)).item()


def calc_Specificity(pred, y, zero_division_safe=False):
    reject = (pred - 1) ** 2
    negatives = (y - 1) ** 2
    if zero_division_safe:
        return (torch.sum(torch.logical_and(reject, negatives)) / (torch.sum(negatives) + 1e-8)).item()
    else:
        return (torch.sum(torch.logical_and(reject, negatives)) / torch.sum(negatives)).item()


if __name__ == '__main__':
    pred = torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]])
    y = torch.tensor([[[0, 0, 0], [1, 1, 0], [1, 1, 0]]])
    print(calc_Specificity(pred, y))
