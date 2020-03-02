from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from nets.darknet53 import darknet_body
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    '''

    Parameters
    ----------
        *args: 
        **kwargs: 
    Returns
    -------

    '''
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   特征层->最后的输出特征层
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    '''

    Parameters
    ----------
        x: 
        num_filters: 
        out_filters: 
    Returns
    -------
        x: 
        y: 

    '''
    # x = compose(
    #         DarknetConv2D_BN_Leaky(num_filters, (1,1)),
    #         DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
    #         DarknetConv2D_BN_Leaky(num_filters, (1,1)),
    #         DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
    #         DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)#最后深度都是512
    # y = compose(
    #         DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
    #         DarknetConv2D(out_filters, (1,1)))(x)
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    # 3个1x1的卷积层（代替全连接层），用于将3个尺度的特征图，转换为3个尺度的预测值。
    y = DarknetConv2D(out_filters, (1,1))(y)
            
    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    '''特征层->最后的输出

    Parameters
    ----------
        inputs: 
        num_anchors: 
        num_classes: 
    Returns
    -------
        Model(inputs, [y1,y2,y3]): 

    '''
    # 生成darknet53的主干模型
    feat1,feat2,feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # 第一个特征层
    # y1=(batch_size,13,13,3,25)
    # 
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            # 执行2倍上采样
            UpSampling2D(2))(x)
    # 将x与darknet的第152层拼接，feat2为第152层输出
    x = Concatenate()([x,feat2])
    # 第二个特征层
    # y2=(batch_size,26,26,3,25)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat1])
    # 第三个特征层
    # y3=(batch_size,52,52,3,25)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    # 根据整个逻辑的输入和输出，构建模型。输入inputs依然保持不变，
    # 即(?, 416, 416, 3)，而输出则转换为3个尺度的预测层，即[y1, y2, y3]
    return Model(inputs, [y1,y2,y3])

#---------------------------------------------------#
#   将预测值的每个特征层输出调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''将预测值的每个特征层调成真实值,
    此处以尺寸为13*13特征层输出为例, num_classes=20

    Parameters
    ----------
        feats: Tensor, <tf.Tensor 'conv2d_59/BiasAdd:0' shape=(?,?,?,75) dtype=float32>
            代表每一个特征层输出,如：(?, 13, 13, 75)
        anchors: array, shape=(3,2), array([[116., 90.],
                                            [156., 198.],
                                            [373., 326.]])，
            代表候选框。            
        num_classes: int, default num_classes=20
        input_shape: Tensor, <tf.Tensor 'mul:0' shape=(2,) dtype=int32>
        calc_loss: bool, 
    Returns
    -------
        box_xy: Tensor, <tf.Tensor 'truediv:0' shape=(?,?,?,3,2) dtype=float32>
            [?,13,13,3,2]
        box_wh: Tensor, <tf.Tensor 'truediv_1:0' shape=(?,?,?,3,2) dtype=float32>
            [?,13,13,3,2]
        box_confidence: Tensor, <tf.Tensor 'sigmoid_1:0' shape=(?,?,?,3,1) dtype=float32>
            [?,13,13,3,1]
        box_class_probs: Tensor, <tf.Tensor 'sigmoid_2:0' shape=(?,?,?,3,1) dtype=float32>
            [?,13,13,3,20]

    '''
    num_anchors = len(anchors)
    
    # anchors_tensor = <tf.Tensor 'Reshape:0' shape=(1,1,1,3,2) dtype=float32>
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)
    
    # grid_shape = <tf.Tensor 'shape:0' shape=(2,) dtype=int32>, eg:(13,13)
    grid_shape = K.shape(feats)[1:3] # height, width

    # grid_y = <tf.Tensor 'Tile:0' shape=(13,13,1,1) dtype=int32>
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    
    # grid_x = <tf.Tensor 'Tile:0' shape=(13,13,1,1) dtype=int32>
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    
    # <tf.Tensor 'concat:0' shape=(13, 13, 1, 2) dtype=int32>
    grid = K.concatenate([grid_x, grid_y])

    grid = K.cast(grid, K.dtype(feats))

    # (batch_size, 13, 13, 3, 25)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # 此处对数值进行了归一化
    # box_xy = <tf.Tensor 'truediv:0' shape=(?,?,?,3,2) dtype=float32>
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    
    # 此处对数值进行了归一化
    # box_wh = <tf.Tensor 'truediv:0' shape=(?,?,?,3,2) dtype=float32>
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    
    # box_confidence = <tf.Tensor 'sigmoid_1:0' shape=(?,?,?,3,1) dtype=float32>
    box_confidence = K.sigmoid(feats[..., 4:5])
    
    # box_class_probs = <tf.Tensor 'sigmoid_2:0' shape=(?,?,?,3,20) dtype=float32>
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子:
#   将box_xy和box_wh的(0~1)相对值，转换为真实坐标，
#   输出boxes是(y_min,x_min,y_max,x_max)的值；
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''

    Parameters
    ----------
        box_xy: Tensor, <tf.Tensor 'strided_slice_15:0' shape=(?,?,?,3,2) dtype=float32>
        box_wh: Tensor, <tf.Tensor 'strided_slice_15:0' shape=(?,?,?,3,2) dtype=float32>
        input_shape: Tensor, <tf.Tensor 'Cast_3:0' shape=(2,) dtype=float32>
        image_shape: Tensor, <tf.Tensor 'Placeholder_366:0' shape=(2,) dtype=float32>
    Returns
    -------
        boxes: Tensor, <tf.Tensor 'Reshape_4:0' shape=(?,4) dtype=float32>

    '''
    # xy和wh取反
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    # 转换类型
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    # offset = <tf.Tensor 'truediv_4:0' shape=(2,) dtype=float32>
    offset = (input_shape-new_shape)/2./input_shape
    # scale = <tf.Tensor 'trundiv_5:0' shape=(2,) dtype=float32>
    scale = input_shape/new_shape
    # <tf.Tensor 'mul_3:0' shape=(?,?,?,3,2) dtype=float32>
    box_yx = (box_yx - offset) * scale
    # <tf.Tensor 'mul_4:0' shape=(?,?,?,3,2) dtype=float32>
    box_hw *= scale
    # 
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    # boxes = <tf.Tensor 'mul_5:0' shape=(?,?,?,3,4) dtype=float32>
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    
    # 
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   获取每个box和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''获取每个box和它的得分

    Parameters
    ----------
        feats: Tensor, <tf.Tensor 'conv2d_59/BiasAdd:0' shape=(?,?,?,75) dtype=float32>
            ,特征层输出(?, 13, 13, 75)
        anchors: array, array([[116., 90.],\n [156., 198.],\n [373.,326.]]), 
        num_classes: int, default num_classes=20
        input_shape: 
        image_shape: 
    Returns
    -------
        boxes:
        box_scores:

    '''
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,20
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    # boxes = <tf.Tensor 'mul_5:0' shape=(?,?,?,3,4) dtype=float32>
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    
    # 获得得分和box
    # eshape，将不同网格的值展平为框的列表，即(?,13,13,3,4)->(?,4)；
    # boxes = <tf.Tensor 'Reshape:0' shape=(?,4) dtype=float32>
    boxes = K.reshape(boxes, [-1, 4])
    
    # box_class_probs = [?,13,13,3,80]
    # box_confidence = [?,13,13,3,1]
    # box_scores为：是框置信度与类别置信度的乘积
    # [?,13,13,3,80] * [?,13,13,3,1] = [?,13,13,3,80]
    box_scores = box_confidence * box_class_probs
    # box_scores: 即(?, 13, 13, 3, 80)->(?, 80)
    box_scores = K.reshape(box_scores, [-1, num_classes])
    
    # boxes:(?, 4), box_scores:(?, 80) 
    return boxes, box_scores

#---------------------------------------------------#
#   图片预测逻辑封装
#   max_boxes：图中最大的检测框数，20个；
#   yolo_outputs输出为：
#   [(?, 13, 13, 255), (?, 26, 26, 255), (?, 52, 52, 255)]
#   anchors列表为：
#   [(10,13), (16,30), (33,23), (30,61), (62,45), (59,119),
#    (116,90), (156,198), (373,326)]
#---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    '''图片预测逻辑封装

    Parameters
    ----------
        yolo_outputs: ndarray, [<tf.Tensor shape=(?, ?, ?, 75) dtype=float32>, 
                                <tf.Tensor shape=(?, ?, ?, 75) dtype=float32>, 
                                <tf.Tensor shape=(?, ?, ?, 75) dtype=float32>]
                    ，三个特征层的输出。
        anchors: 
        num_classes: 
        image_shape: 
        max_boxes: int, 表示一张图片中最多的框数，此处默认为20
        score_threshold: 
        iou_threshold: 
    Returns
    -------
        boxes_: 
        scores_: 
        classes_: 

    '''
    # 获得特征层的数量
    num_layers = len(yolo_outputs)
    # 第一特征层13*13对应的anchor是678
    # 第二特征层26*26对应的anchor是345
    # 第三特征层52*52对应的anchor是012
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    # 输入图像的尺寸，也就是第0个特征图的尺寸乘以32，即13x32=416，
    # 这与Darknet的网络结构有关。
    # input_shape = <tf.Tensor 'mul:0' shape=(2,) dtype=int32>
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
        # boxes: (?, 4)  # ?是框数
        # box_scores: (?, 80)
    
    # 将每个特征层的结果进行堆叠，
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    # max_boxes_tensor，每张图片的最大检测框数，max_boxes是20；
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 通过掩码mask和类别c，筛选出：框class_boxes和置信度class_box_scores
        # 取出所有box_scores >= score_threshold的框，和置信度得分
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些
        # 通过NMS，非极大值抑制，筛选出框boxes的NMS索引nms_index；
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，置信度得分与类别
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # boxes_ = <tf.Tensor 'concat_11:0' shape=(?,4) dtype=float32>
    boxes_ = K.concatenate(boxes_, axis=0)
    # scores_ = <tf.Tensor 'concat_12:0' shape=(?,) dtype=float32>
    scores_ = K.concatenate(scores_, axis=0)
    # classes_ = = <tf.Tensor 'concat_13:0' shape=(?,) dtype=float32>
    classes_ = K.concatenate(classes_, axis=0)

    # 将多个类别的数据组合，生成最终的检测数据框，并返回
    # 输出格式:
    # boxes_: (?, 4)
    # scores_: (?,)
    # classes_: (?,)
    return boxes_, scores_, classes_


