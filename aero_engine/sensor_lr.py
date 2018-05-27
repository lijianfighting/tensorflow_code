# -*- coding: utf-8 -*-

# logRegression: Logistic Regression with SGD ，和sensor_convolutional_mlp.py一套
# Author : jimmy lee 
# Date   : 2018-1-5
# 传感器故障诊断 

from __future__ import print_function

__docformat__ = 'restructedtext en'

#import cPickle as pickle
#import pickle
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
from sklearn import decomposition #pca

import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import pylab as pl

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)  #softmax是sigmoid的扩展，可用于多分类

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)   #模型预测值，书p17
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]  #模型参数，训练出最佳参数

        # keep track of model input
        self.input = input
    #用MSGD法计算误差
    def negative_log_likelihood(self, y):   # NLL损失函数 p20
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):   #判断样本是否分类错误
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label  # y是正确的标签
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y)) #T.neq 分类错误返回1,总的返回错误率
        else:
            raise NotImplementedError() #类型错误，不是int
    #自己加的，测试
    def y_predict(self):
    	return self.y_pred
#以上为LR类


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print('... loading data')

    # Load the dataset
    # with gzip.open(dataset, 'rb') as f:
    #     try:
    #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    #     except:
    #         train_set, valid_set, test_set = pickle.load(f)
    fr = open(dataset, 'rb')    #open的参数是pkl文件的路径
    #fr = open('data_sensor.pkl')    #open的参数是pkl文件的路径
    inf = pickle.load(fr)       #读取pkl文件的内容
    fr.close()   
    train_set,test_set = inf
    valid_set = test_set

    train_set[0] = train_set[0][:,:256]  #	取306个特征的前16*16=256个特征，CNN用
    test_set[0] = test_set[0][:,:256]
    valid_set[0] = valid_set[0][:,:256]

    # train_set[0] = train_set[0][:,50:]  #  取306个特征的前16*16=256个特征，CNN用
    # test_set[0] = test_set[0][:,50:]
    # valid_set[0] = valid_set[0][:,50:]

    pca = decomposition.PCA(n_components=196)        #数据进行PCA降维处理
    train_set[0]  = pca.fit_transform(train_set[0])
    valid_set[0]  = pca.fit_transform(valid_set[0])
    test_set[0]  = pca.fit_transform(test_set[0])

    print(numpy.shape(train_set[0]),numpy.shape(train_set[1]))

    # train_set[0] = train_set[0][300:,:]  #去掉故障0数据
    # test_set[0] = test_set[0][200:,:]
    # #valid_set[0] = valid_set[0][200:,:] #valid_set和test_set是引用关系，只改一个就行

    # train_set[1] = train_set[1][300:]  
    # test_set[1] = test_set[1][200:]
    # #valid_set[1] = valid_set[1][0:]


    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):  #数据集导入到共享变量中（为了GPU运行）
        """ Function that loads the dataset into shared variables 

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]  #返回的数组
    return rval   #返回结果


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='data_sensor.pkl',
                           batch_size=600):   #训练集50000个样本，50000/600 = 83
    """  #随机梯度法SGD计算模型参数
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)  # dataset='mnist.pkl.gz'

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #  建立模型
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch  第index个batch
    #p22
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images 栅格图像
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=256, n_out=7)  #分类器为LR

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)  #代价函数，损失函数，为NLL

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch  #p24   测试集中分类错误的样本
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size], #1个batch1一个batch地测试
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(   #验证集中分类错误的样本
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b) #p22
    g_W = T.grad(cost=cost, wrt=classifier.W)  #求损失函数的梯度
    g_b = T.grad(cost=cost, wrt=classifier.b)  #第一项对第二项求导

    #p22
    # start-snippet-3 
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.               #用梯度下降法更新参数W、b
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`  #计算并返回每个minibatch的cost，同时执行MSGD，更新模型参数W、b
    train_model = theano.function(
        inputs=[index],
        outputs=cost,     #输出为cost
        updates=updates,  #更新参数（与验证、测试模型不同之处）
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL # 训练模型
    ###############
    print('... training the model')
    # early-stopping parameters  提前停止的参数 p13   
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)  #n_train_batches line287
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):  #一个batch一个batch地训练

            minibatch_avg_cost = train_model(minibatch_index)  #训练模型，返回cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)   #验证集上分类错误的样本
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss #更新最好验证集准确度
                    # test it on the test set

                    test_losses = [test_model(i)  #测试集上分类错误的样本
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model  保存最佳参数
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True  #达到要求，停止训练 （不知此处是不是基于：如果在验证集上的性能不能增加，则提前停止训练）
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    #predict()
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    # We can test it on some examples from test test
    dataset='data_sensor.pkl'
    #datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:])
    print("Predicted values for the first 20 examples in test set:")
    #print(predicted_values)
    x1 = range(1200)
    y1 = test_set_y #出错
    y2 = predicted_values
    print(numpy.shape(y2))

    #pl.plot(x1, y1)# use pylab to plot x and y
    pl.plot(x1, y2)
    pl.show()# show the plot on the screen


def predict():  #用之前训练好的模型测试新样本
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    #classifier = pickle.load(open('/Users/lijian/Desktop/研究生/深度学习/code/chapter4/best_model.pkl'))
    classifier = pickle.load(open('best_model.pkl'))  #路径不要包含汉字

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    # fr = open('data_sensor.pkl')    #open的参数是pkl文件的路径
    # inf = pickle.load(fr)       #读取pkl文件的内容
    # fr.close()   
    # train_set,test_set = inf
    # test_set_x = test_set[0]
    # test_set_y = test_set[1]
    # print("shape of test_set_x:",numpy.shape(test_set_x))
    # print(numpy.shape(train_set[0]))

    # We can test it on some examples from test test
    dataset='data_sensor.pkl'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    print("shape of test_set_x:",numpy.shape(test_set_x))

    # train_set_x, train_set_y = datasets[0] #test
    # train_set_x = train_set_x.get_value()
    # print("shape of train_set_x:",numpy.shape(train_set_x))
    predicted_values = predict_model(test_set_x[:])
    print("Predicted values for the first 20 examples in test set:")
    #print(predicted_values)
    x1 = range(1200)
    y1 = test_set_y #出错
    y2 = predicted_values
    print(numpy.shape(x1))
    print(numpy.shape(y1))
    print(numpy.shape(y2))
    #pl.plot(x1, y1)# use pylab to plot x and y
    pl.plot(x1, y2)
    pl.show()# show the plot on the screen



if __name__ == '__main__':
    sgd_optimization_mnist()
    #predict()


