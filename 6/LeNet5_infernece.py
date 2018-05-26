
# coding: utf-8
# 和LeNet5_train.py为一套程序
import tensorflow as tf


#1. 设定神经网络的参数

INPUT_NODE = 784   #28*28
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1  #28*28*1
NUM_LABELS = 10

CONV1_DEEP = 32  #32个卷积核
CONV1_SIZE = 5   # 5*5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512 #全连接层结点数


#2. 定义前向传播的过程
def inference(input_tensor, train, regularizer):
	#第一层：卷积层，过滤器的尺寸为5×5，使用零填充，输出28*28*32
    with tf.variable_scope('layer1-conv1'): #命名空间layer1-conv1,不需要担心变量重名了
        conv1_weights = tf.get_variable( #通过tf.get_variable的方式创建变量（过滤器的权重变量和偏置项变量）
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], #5，5，1，32, 分别为过滤器尺寸、当前层的深度、过滤器深度（个数），本层参数个数5*5*1*32+32
            initializer=tf.truncated_normal_initializer(stddev=0.1)) #初始化为满足正态分布的随机值，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        #tf.nn.conv2d实现卷积运算，input_tensor要求为四维矩阵，conv1_weights为卷积层的权重，strides中第2、3个数为在数据的长、宽上的步长，其他两个数固定为1
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') #卷积运算，步长1，零填充，输出还是28*28
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases)) #卷积后经过激活函数，去线性化

    #第二层：池化层，滤波器2*2，步长2，零填充，输出14*14*32
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME") #池化运算，滤波器2*2，步长2，零填充，输出14*14*32
     
    #第三层：卷积层，过滤器的尺寸为5×5，使用零填充，#输出14*14*64                                                                                   #池化层没有参数
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],#5，5，32，64
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #输出14*14*64
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    #第四层：池化层，滤波器2*2，步长2，零填充，输出7*7*64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#输出7*7*64
        pool_shape = pool2.get_shape().as_list()

        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] #矩阵拉成向量 7*7*64
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) #pool_shape[0]是一个batch中数据的个数

    #第五层：全连接层，输入7*7*64，输出512*1的向量
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],#输入7*7*64，输出512，
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights)) #只有全连接层的权重需要加入正则化
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases) #全连接层运算
        if train: fc1 = tf.nn.dropout(fc1, 0.5) #dropout一般只在全连接层而不是卷积池化层使用

    #第六层：全连接层，输入512*1，输出10*1的向量，这一层的输出通过softmax之后就能得到最后的分类结果了
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS], #权重矩阵w的维度
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases #第六层的输出

    return logit #返回第六层的输出（没有经过第七层的softmax）





