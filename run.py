# -*- coding: utf-8 -*-
from scripts.style_transfer import transfer

# 调参是个技术活
parameters = {
    'content_image' : 'images/tubingen.jpg',
    'style_image' : 'images/starry_night.jpg',
    'image_size' : 192, # 等比例压缩，短边至该尺寸，提速但会降低输出图片的分辨率
    'style_size' : 192, # 压缩提取风格的图片，可以适当压缩提速
    'content_layer' : 3, # 0到12共13层可选
    'content_weight' : 6e-2, # 内容的权重
    'style_layers' : [1, 4, 6, 7], # 0到12共13层可选
    'style_weights' : [300000, 1000, 15, 3],
    'tv_weight' : 2e-2, # 让图像更平滑
    'initial_lr' : 3.0,
    'decayed_lr' : 0.1,
    'decay_lr_at' : 180,
    'max_iter' : 200
}

transfer(**parameters)