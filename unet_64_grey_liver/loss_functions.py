import torch


def loss_func():
    return torch.nn.BCELoss()

# https://blog.csdn.net/baidu_36511315/article/details/105217674
def diceCoeff(pred, gt, smooth=1e-5):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    bs = gt.size(0)
    pred_flat = pred.view(bs, -1)
    gt_flat = gt.view(bs, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
    return loss.sum() / bs


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()  # 反向传播产生的是每个参数对应的梯度
        opt.step()  # 这步会用梯度和学习率更新参数，用这函数会自动避免把更新参数的计算步骤加入计算图
        opt.zero_grad()

    return loss.item(), diceCoeff(model(xb), yb), len(xb)
