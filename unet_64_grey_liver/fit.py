import numpy as np
import loss_functions
import torch
import matplotlib.pyplot as plt


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    val_losses=[]
    val_dice_losses=[]
    for epoch in range(epochs):
        # 训练，启用 BatchNormalization 和 Dropout。这两个本来只是后提出来的用来提升网络训练效果的算法组件，
        # 但是由于太好使太通用了，就被整合进网络中。但是这俩组件是在训练时使用的，所以训练前要显式打开，验证前要显式关闭。
        model.train()
        for xb, yb in train_dl:
            _, _, _ = loss_functions.loss_batch(model, loss_func, xb, yb, opt)

        # 验证，不启用 BatchNormalization 和 Dropout
        model.eval()
        with torch.no_grad():
            losses, dice_losses, nums = zip(
                *[loss_functions.loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_dice_loss = np.sum(np.multiply(dice_losses, nums)) / np.sum(nums)

        print('\nepoch:', epoch, '\nval_crossentropy_loss:', val_loss,'\nval_dice_coe:', val_dice_loss)
        
        val_losses.append(val_loss)
        val_dice_losses.append(val_dice_loss)
    
    # https://www.jianshu.com/p/82b2a4f66ed7
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    x = np.array(range(epochs))
    y_1 = np.array(val_losses)
    y_2 = np.array(val_dice_losses)
    ax1.plot(x, y_1, 'c*-', label='ax1', linewidth=1)
    ax2.plot(x, y_2, 'm.-.', label='ax2', linewidth=1)

    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('binary cross entropy')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Dice coeff')

    plt.show()

