from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # L2正则化
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    # Padding，一般使用same模式，只有当步长为(2,2)时，使用valid模式。避免在降采样中，引入无用的边界信息
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   残差结构块：由下采样卷积网络组成
#   DarknetConv2D + BatchNormalization + LeakyReLU
#   darknet53中的残差卷积就是进行一次3X3卷积核、步长为2的
#   卷积，然后保存该卷积layer，再进行一次1X1的卷积和一次
#   3X3的卷积，并把这个结果加上layer作为最后的结果。
#   
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    # 以x = resblock_body(x, 64, 1)调用为例：
    # 填充x的边界为0，由(?, 416, 416, 32)转换为(?, 417, 417, 32)。
    # 因为下一步卷积操作的步长为2，所以图的边长需要是奇数
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    # DarkNet的2维卷积操作，核是(3,3)，步长是(2,2)，注意，这会导致特征尺寸变小，
    # 由(?, 417, 417, 32)转换为(?, 208, 208, 64)。由于num_filters是64，
    # 所以产生64个通道。
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        # 残差（Residual）操作，将x的值与y的值相加。残差操作可以避免，
        # 在网络较深时所产生的梯度弥散问题（Vanishing Gradient Problem）。
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   残差网络darknet 主体部分由52个卷积层组成
#---------------------------------------------------#
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    # 1次卷积，共1次
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    
    # 下面进行5次步长为2的卷积，降采样
    # 3次卷积，共4次
    x = resblock_body(x, 64, 1)
    # 6次卷积，共10次
    x = resblock_body(x, 128, 2)
    # 24次卷积，共34次
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 24次卷积，共计58次
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 12次卷积，共计70次
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

