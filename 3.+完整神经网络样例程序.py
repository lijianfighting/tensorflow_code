
# coding: utf-8
import tensorflow as tf
from numpy.random import RandomState

# #### 1. 定义神经网络的参数，输入和输出节点。
batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# #### 2. 定义前向传播过程，损失函数及反向传播算法。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2) #没有激活函数，是个线性模型
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) #y的值限制在1e-10～1之间 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


# ####  3. 生成模拟数据集。
rdm = RandomState(1)
X = rdm.rand(128,2) #data_size=128
Y = [[int(x1+x2 < 1)] for (x1, x2) in X] #样本标签，这里x1+x2<1的样例都被认为是正样本

# #### 4. 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() #初始化所有变量W1、W2、X
    sess.run(init_op)
    
    # 输出目前（未经训练）的参数取值。
    print "w1:", sess.run(w1)
    print "w2:", sess.run(w2)#初始化权重
    print "\n"
    
    # 训练模型。
    STEPS = 5000 #训练5000次，每次一个batch
    for i in range(STEPS):
        start = (i*batch_size) % 128 #128是data_size，batch_size = 8
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]}) #运行train_step，，每遍历batch_size个数就更新一次参数
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y}) #运行计算交叉熵
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print "\n"
    print "w1:", sess.run(w1)
    print "w2:", sess.run(w2) #训练完后的权重




