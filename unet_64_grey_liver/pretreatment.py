from pathlib import Path
import torch.utils.data as data
import PIL.Image as Image
import os
from torchvision.transforms import transforms

# 指定目标路径
DATA_PATH = Path('liver')
TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "val"


# 构建数据路径的列表
def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)  # 这不是取模运算，这是python的格式化输入
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))  # 这个imgs是存所有图片路径的list，存的不是图片本身。原图和对应掩模一个元组。
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform  # 函数变量，把Image.open的数据格式转换成网络能用的数据格式，对象是训练集。
        self.target_transform = target_transform  # 上面那行是转换训练集，这行是转换验证集。

    # origin_x是PIL里定义的一个类的实例，可以用torchcision.transforms.ToTensor()转成tensor
    # 对于本例的RGB三通道输入来说，每一个origin_x在转完的tensor里的结构还是三个通道的数据依次储存，如下所示：
    '''
    tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.]],
    
            [[0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.]],
    
            [[0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.]]])
    '''
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        origin_x = Image.open(x_path).convert('L')  # 还是直接转单通道灰度图吧
        origin_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(origin_x)
        if self.target_transform is not None:
            img_y = self.target_transform(origin_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


def result(bs):
    # 函数变量，而且还把两个函数调用合成一个了，用于把PIL类的图像实例转化成tensor并normalize，这里的对象是原图像。
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # 函数变量，mask只需要转换为tensor，这里的对象是原图像对应的掩模。
    y_transforms = transforms.ToTensor()

    train_dataset = LiverDataset(TRAIN_PATH.as_posix(), transform=x_transforms, target_transform=y_transforms)
    val_dataset = LiverDataset(VAL_PATH.as_posix(), transform=x_transforms, target_transform=y_transforms)

    from torch.utils.data import DataLoader

    # DataLoader的输入不一定要Dataset子类的实例，满足__getitem__()和__len__()协议的数据集也行，
    # 例如，当使用dataset[idx]访问时，此类数据集可以从磁盘上的文件夹中读取第idx张图像及其对应的标签。
    # 详情见：https://pytorch.apachecn.org/docs/1.4/96.html
    train_loaders = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loaders = DataLoader(val_dataset, batch_size=bs)

    return train_loaders, val_loaders
