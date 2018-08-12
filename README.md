# 图像风格转移

复现论文[Image Style Transfer Using Convolutional Neural Networks" (Gatys et al., CVPR 2016)](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

输入两张图片，把一张图片的风格转移到另一张图片上。

受限于 GPU 性能，选择轻量化的 SqueezeNet 网络，分别构造内容和风格的损失函数，在网络的不同层上进行梯度下降。并添加总变化损失，使图像更加平滑。

效果

![内容图片](https://raw.githubusercontent.com/Xingyb14/My_image_hosting_site/master/content.jpg)

![风格图片](https://raw.githubusercontent.com/Xingyb14/My_image_hosting_site/master/style.jpg)

![输出图片](https://raw.githubusercontent.com/Xingyb14/My_image_hosting_site/master/transfered.jpg)

