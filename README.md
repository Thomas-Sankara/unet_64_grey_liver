# unet_64_grey_liver
参考了很多博客也犯了不少低级错误。<br>
比如我是从之前的mnist_nn_separate改写的，但是忘改学习率了。<br>
一般来说，CNN的学习率是0.00001，以此为准再放大或缩小10倍。但之前那个因为特征好学，不太敏感，学习率就是0.1，结果代码复用时忘改了。<br>
再比如损失函数，你要是自己在网络最后把输出用sigmoid处理了，就别调用内置sigmoid的损失函数了，实现DICE系数时也是一样。<br>
环境什么的不多说了，和mnist_nn_separate一样。<br>
train数据集要自己解压，四个压缩包解压到一个文件夹里，最后目录结构应该和val一样，看看代码也能明白。<br>

![image](https://github.com/charcurse/unet_64_grey_liver/blob/master/images/correct_unet_64_grey.png)
