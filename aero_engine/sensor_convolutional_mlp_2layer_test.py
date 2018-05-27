#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cnn ,with sensor_lr
# Author : jimmy lee 
# Date   : 2018-1-6
# 传感器故障诊断 

from __future__ import print_function

import os
import sys
import timeit

import numpy
from sklearn import decomposition #pca
import six.moves.cPickle as pickle

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import matplotlib.pyplot as plt
import pylab as pl

from sensor_lr import LogisticRegression, load_data
from sensor_mlp import HiddenLayer

# implements a {convolution + max-pooling} layer.
class LeNetConvPoolLayer(object):   #卷积+下采样
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])  #prod函数求积
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(                                                 #初始化W
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)  #初始化b
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters  用过滤器卷积输入的特征映射
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling  用maxpooling下采样每个特征映射
        pooled_out = pool.pool_2d(
            input=conv_out,   #卷积输出作为输入
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        #self.output = theano.tensor.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  #卷积+下采样后的输出（经过激活函数）
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  #卷积+下采样后的输出（经过激活函数）
        
        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

# start-snippet-2  自己写的
class LeNet_test(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    #def __init__(self, rng, input, n_in, n_hidden, n_out, nkerns, batch_size):
    def __init__(self, rng, input, n_out, nkerns, batch_size,n_hidden,image_shape,filter_shape,poolsize):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.layer0_input = input.reshape((batch_size, 1, image_shape[0], image_shape[0])) #转成16*16的图片
        #layer0_input = x.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)    第一层

        image_shape[1] = (image_shape[0] - filter_shape[0] + 1) / poolsize[0] #计算经过卷积+池化后图像的尺寸
        image_shape[2] = (image_shape[1] - filter_shape[1] + 1) / poolsize[1]
        print('image shape:',image_shape)

        self.layer0 = LeNetConvPoolLayer(          
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, 1, image_shape[0], image_shape[0]), #（batch_size,输入特征映射数量，图像高，图像宽）
            filter_shape=(nkerns[0], 1, filter_shape[0], filter_shape[0]),   #（过滤器数量，输入特征向量数量，过滤器高，过滤器宽）
            poolsize=(poolsize[0], poolsize[0]) #下采样参数（行，列）
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)      第二层
        self.layer1 = LeNetConvPoolLayer(
           rng,
           input=self.layer0.output,  #上层输入作为下层输入
           image_shape=(batch_size, nkerns[0], image_shape[1], image_shape[1]),   #输入6*6*20
           filter_shape=(nkerns[1], nkerns[0], filter_shape[1], filter_shape[1]),
           poolsize=(poolsize[1], poolsize[1])
        )                                                   #输出1*1*50

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer （隐层+输出层）            第三层（隐层）
        self.layer2 = HiddenLayer(
           rng,
           input=self.layer2_input,
           n_in=nkerns[1] * image_shape[2] * image_shape[2], #隐层输入单元数
           n_out=n_hidden,
           #activation=T.tanh
           activation=theano.tensor.nnet.relu
       
        )

        # classify the values of the fully-connected sigmoidal layer           第四层（输出层）
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=n_hidden, n_out=7) 

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.layer0.W).sum()              #L1正则化
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()  
            + abs(self.layer3.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.layer0.W ** 2).sum()            #L2正则化
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
            + (self.layer3.W ** 2).sum()
        )


        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (               #NLL损失函数
            self.layer3.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.layer3.errors  #分类错误的样本数

        #自己加的，测试
        self.y_predict = self.layer3.y_predict #预测值

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.layer0.params + self.layer1.params + self.layer2.params + self.layer3.params  #两层参数都是self.params = [self.W, self.b]
        # end-snippet-3

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=1000, 
                    L1_reg=0.00, L2_reg=0.001,
                    image_shape=[14,5,1], filter_shape=[5,5], poolsize=[2,1],
                    dataset='data_sensor.pkl',
                    nkerns=[10, 20], batch_size=700,n_hidden=500):   
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer   卷积核（过滤器）数量
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # train_set_x = train_set_x.get_value()  #test
    # train_set_x = train_set_x[:,:288]
    # print(numpy.shape(train_set_x))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized（栅格化） images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # construct the MLP class  分类器为MLP
    classifier = LeNet_test(
        rng=rng,
        input=x,
        n_out=7, #故障种类
        nkerns=nkerns,
        batch_size=batch_size,
        n_hidden=n_hidden,
        image_shape=image_shape,
        filter_shape=filter_shape,
        poolsize=poolsize,

    )


    # the cost we minimize during training is the NLL of the model
    #cost = classifier.negative_log_likelihood(y)   #代价函数NLL
    cost = (
        classifier.negative_log_likelihood(y)   #代价函数=NLL+正则化项
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #自己加的，训练集误差
    train_model_error = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    # create a list of all model parameters to be fit by gradient descent
    params = classifier.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)      #梯度下降法更新参数
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1
    predict_model = theano.function(
        inputs=[index],
        #outputs=classifier.layer3.y_pred,
        outputs=classifier.y_predict(),
        #updates=updates, #这句去掉，下面givens里面的y也要去掉，不然报错（测试的时候，不用更新参数了，只有训练的时候需要）
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            #y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    validation_loss_plot = [] #定义画图列表
    train_loss_plot = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index) #当前的cost
            # cost_ij = numpy.mean(cost_ij)
            # print('epoch %i,train cost %f' %(epoch, cost_ij)) 

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)  #此处验证集和测试集貌似并没有影响训练的参数，只是达到一个提前停止的作用
                print('epoch %i, minibatch %i/%i, validation error %f %%' %   
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                validation_loss_plot.append(this_validation_loss) #验证集误差画图列表

                #训练集误差，自己加的
                train_losses = [
                    train_model_error(i)
                    for i in range(n_test_batches)
                ]
                train_score = numpy.mean(train_losses)
                if train_score != 0:  #训练集误差降到0后就不打印了
                    print('epoch %i, minibatch %i/%i, train error %f %%' %   
                        (epoch, minibatch_index + 1, n_train_batches,
                        train_score * 100.))

                train_loss_plot.append(train_score) #验证集误差画图列表

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:    

                    #improve patience if loss improvement is good enough  #如果验证集误差减小了千分之五，则更新最小误差
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)  #控制提前停止的变量

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss #更新最小误差
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    # #save the best model  保存最佳参数
                    # with open('best_model_LeNet_test.pkl', 'wb') as f:
                    #     pickle.dump(classifier, f)  #不对

            if patience <= iter:
                done_looping = True #提前停止
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    epoch_times = 0
    x0 = range(n_epochs)

    #plt.subplot(211)
    pl.figure(1)
    pl.plot(x0, validation_loss_plot, 'r',label="validation error") #验证集误差
    pl.plot(x0, train_loss_plot, 'g',label="train error")  #训练集误差
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('error rate')
    plt.title('The training process of PCA-CNN')


    # We can test it on some examples from test test
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_x = test_set_x[:,:256]

    print("shape of test_set_x:",numpy.shape(test_set_x))
    print("type of test_set_y:",type(test_set_y))    

    fr = open(dataset, 'rb')   
    train_set,test_set = pickle.load(fr)  #为了画出实际标签，再次导入数据

    predicted_values = numpy.array([predict_model(i) for i in range(n_test_batches)]) #列表转成数组
    predicted_values = predicted_values.reshape(n_test_batches * batch_size) #转成1维的数组

    #print("type of predicted_values:",type(predicted_values))    
    #print("shape of predicted_values:",numpy.shape(predicted_values))

    print("Predicted values for test set:")
    x1 = range(test_set_x.shape[0])
    x2 = range(n_test_batches * batch_size) #如果样本数不能整除batch_size,则剩下的不能遍历到

    y1 = test_set[1][:n_test_batches * batch_size] #实际标签y（前n_test_batches * batch_size个）
    y2 = predicted_values

    predict_error_value = []
    for i in range(n_test_batches * batch_size):
        if y1[i] != y2[i]:
            predict_error_value.append(i) #预测错误的样本序号

    print("Predicted error values for test set:\n", predict_error_value)
    print("The number of Predicted error values :", numpy.shape(predict_error_value))

    #plt.subplot(212)
    pl.figure(2)
    plt.plot(x2, y1,'g.', markersize=1,linewidth=1.0,label="real values")  #实际值
    plt.plot(x2, y2,'r.', markersize=1,linewidth=1.0,label="predict values")  #预测值
    plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('type of fault')
    plt.title('The results of classification')
    pl.show()# show the plot on the screen


    return validation_loss_plot[-1] #返回最终的验证集误差



def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)


if __name__ == '__main__':
    #evaluate_lenet5()
    #predict()
    
    evaluate_lenet5(learning_rate=0.1, n_epochs=1500, 
                        L1_reg=0.00, L2_reg=0.001,
                       image_shape=[14,2,2], filter_shape=[5,3], poolsize=[2,2],
                       dataset='data_sensor.pkl',
                       nkerns=[10, 30], batch_size=700,n_hidden=500)

#没有降维时不要正则化，降维的正则化系数设为0.001





