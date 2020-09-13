import torch
import pretreatment  # 同目录下自己定义的数据预处理文件
import model  # 同目录下的自己定义的网络结构文件
from torch import optim
import loss_functions  # 相同目录下自己定义的损失函数文件
import fit  # 同目录下自己定义的运行训练和验证的文件

# 超参数
bs = 8  # batch size
lr = 0.00001  # learning rate
epochs = 50  # how many epochs to train for

# 加载处理好的数据
train_dl, valid_dl = pretreatment.result(bs)

# 下面就是要在GPU上跑代码了，教程是接着sequential写法写的，我找到适用于常规写法的后续代码了，
# 见：https://zhuanlan.zhihu.com/p/61875829 有意思的是，其实常规写法也定义了数据封装函数和类了，绕不过去。
print(torch.cuda.is_available())

# 创建设备对象
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 定义一个可将数据移动到所选设备的函数
def to_device(data, device):
    if isinstance(data, (list, tuple)):  # 对于数据集，就调用这行了
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)  # 对于model，就调用这行了


# 封装我们已有的数据加载器并在读取数据批时将数据移动到所选设备。我们不需要扩展已有的类来创建 PyTorch 数据加载器。
# 我们只需要用 __iter__ 方法来检索数据批并使用 __len__ 方法来获取批数量即可。
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# 把数据集装入显存中并用GPU做后续运算
train_dl = DeviceDataLoader(train_dl, dev)
valid_dl = DeviceDataLoader(valid_dl, dev)

model = model.Unet(1, 1)  # 这行代码是用CNN模型训练的
to_device(model, dev)  # 把模型及其参数也都放到显存里，再用GPU运行

loss_func = loss_functions.loss_func()

# 选择优化器
opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

fit.fit(epochs, model, loss_func, opt, train_dl, valid_dl)
