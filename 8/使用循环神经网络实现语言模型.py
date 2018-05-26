
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import reader


# #### 1. 定义相关的参数。

DATA_PATH = "../datasets/PTB_data"
HIDDEN_SIZE = 200 #隐层规模，词向量维度
NUM_LAYERS = 2 #LSTM结构的层数
VOCAB_SIZE = 10000 #词典规模，单词数

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35 #训练数据截断长度 (防止梯度消失)

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1 #测试数据截断长度（测试时数据不需要截断）
NUM_EPOCH = 2
KEEP_PROB = 0.5 #节点不被dropout的概率
MAX_GRAD_NORM = 5 #控制梯度膨胀的参数 (防止梯度爆炸)


# #### 2. 定义一个类来描述模型结构，这样方便维护循环神经网络中的状态。

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # 定义使用LSTM结构为循环体结构，及训练时使用dropout。 
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE) # 括号中为隐藏层规模
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS) ## 2层LSTM结构
        
        # 初始化最初的状态。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])#get_variable如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的
        # embedding 词向量，每个单词向量的维度为：HIDDEN_SIZE
        # 将原本单词ID转为单词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if is_training: #只在训练时使用dropout
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。先将不同时刻LSTM结构的输出收集起来，再通过一个全连接层得到最终的输出
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state) #从输入数据中获取当前时刻的输入+上一时刻状态，传入LSTM结构
                outputs.append(cell_output) #先将不同时刻LSTM结构的输出收集起来
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE]) 
        bias = tf.get_variable("bias", [VOCAB_SIZE])#LSTM的输出再经过一个全连接层
        logits = tf.matmul(output, weight) + bias   #得到最后的预测值
        
        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], #预测值
            [tf.reshape(self.targets, [-1])], #实际值，压成一维数组
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])#损失权重，这里所有权重都为1，也就是说不同batch和时刻的重要程度一样
        self.cost = tf.reduce_sum(loss) / batch_size #计算每个batch的平均损失
        self.final_state = state
        
        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，避免梯度爆炸 
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        #定义优化方法和训练步骤。
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# #### 3. 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值

def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch。
    for step in range(epoch_size):
        x, y = session.run(data)
        #在当前batch上运行train_op并计算损失值，交叉熵损失函数计算的就是下一个单词为给定单词的概率
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters) #返回给定模型在给定数据上的perplexity值（复杂度，越小越好，为n表示下个单词从n中选一个）


# #### 4. 定义主函数并执行。

def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE #batch大小
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP #数据截断长度

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP
    
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    #定义训练用的RNN模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
        
    #定义评测用的RNN模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        ## 通过队列依次读取batch
        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            #在所有数据上训练RNN模型
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)#训练

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)#验证
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()





