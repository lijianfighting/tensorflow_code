
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece # 和LeNet5_infernece.py为一套程序
import os
import numpy as np

#1. 定义神经网络相关的参数
BATCH_SIZE = 1 #一个batch中有100个数据（自己改一下试试）
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99 #学习率衰减
REGULARIZATION_RATE = 0.0001 #正则化系数
TRAINING_STEPS = 6000 #一个step训练1个batch的样本
MOVING_AVERAGE_DECAY = 0.99 #滑动平均


# #### 2. 定义训练过程

def train(mnist):
    # 定义输入为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],#输入的数据格式是一个四维矩阵
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input') #尺寸中OUTPUT_NODE为10，one-hot向量
    #正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer) #前向传播得到预测值y（不经过softmax的向量）， train = False
    #全局步数
    global_step = tf.Variable(0, trainable=False)#global_step在滑动平均、优化器、指数衰减学习率等方面都有用到，初始值设为0，系统会自动更新这个参数的值
                                                 #代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表
    # 定义滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 定义损失函数，计算交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) #tf.argmax(y_, 1)返回y_中最大值的索引号，即真实数字
    cross_entropy_mean = tf.reduce_mean(cross_entropy) #reduce_mean计算平均值
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,  #学习率和全局步数有关
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) #梯度下降法
    # 每过一遍数据即要通过反向传播来更新网络参数，又要更新每一个参数的滑动平均值，为了一次完成多个操作，使用如下机制
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess: #生成会话
        tf.global_variables_initializer().run() #初始化所有变量
        for i in range(TRAINING_STEPS): #一个step训练1个batch的样本
            xs, ys = mnist.train.next_batch(BATCH_SIZE) #一次取一个batch的数据

            reshaped_xs = np.reshape(xs, (    #输入数据格式，100，28，28，1
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE, 
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) #运行train_op，返回loss, global_step（貌似）

            if i % 1000 == 0: #每1000个TRAINING_STEPS打印一次
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


# #### 3. 主程序入口

def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True) # 专门处理minist的，会自动将mnist数据集划分为训练、验证、测试集
    train(mnist)                                                              # 注意：和lenet_5_test.py 中使用的数据不一样

if __name__ == '__main__':
    main()

#20180526
# After 1 training step(s), loss on training batch is 5.34701.
# After 1001 training step(s), loss on training batch is 0.695222.
# After 2001 training step(s), loss on training batch is 0.80559.
# After 3001 training step(s), loss on training batch is 0.650085.
# After 4001 training step(s), loss on training batch is 0.655199.
# After 5001 training step(s), loss on training batch is 0.633183.
# [Finished in 1741.4s]

