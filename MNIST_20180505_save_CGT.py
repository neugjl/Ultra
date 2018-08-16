import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.training import moving_averages

#读取训练、验证、测试数据
train_data_raw=pd.read_table('./train_validation_test/train_data.txt',header=None)
validation_data_raw=pd.read_table('./train_validation_test/validation_data.txt',header=None)
test_data_raw=pd.read_table('./train_validation_test/test_data.txt',header=None)
train_label_raw=pd.read_table('./train_validation_test/train_label_MNISTformat.txt',header=None)
validation_label_raw=pd.read_table('./train_validation_test/validation_label_MNISTformat.txt',header=None)
test_label_raw=pd.read_table('./train_validation_test/test_label_MNISTformat.txt',header=None)

train_data=train_data_raw.values
validation_data=validation_data_raw.values
test_data=test_data_raw.values
train_label=train_label_raw.values
validation_label=validation_label_raw.values
test_label=test_label_raw.values


INPUT_NODE =  57                                                               #输入节点
OUTPUT_NODE = 3                                                                #输出节点
LAYER1_NODE = 60                                                               #第1隐藏层节点   
LAYER2_NODE = 30                                                               #第2隐藏层节点                      
BATCH_SIZE =16384                                                              #每次batch的样本个数      
BATCH_SIZE =4096   

#模型相关的参数
LEARNING_RATE_BASE = 0.8                                                       #设置指数衰减的学习率，见《Tensorflow 实战Google深度学习框架（第一版）》p85
LEARNING_RATE_DECAY = 0.99                                                     
REGULARAZTION_RATE = 0.0001                                                    #正则化项权重，见《Tensorflow 实战Google深度学习框架（第一版）》p88
TRAINING_STEPS = 300000                                                        #总迭代轮数
MOVING_AVERAGE_DECAY = 0.99                                                    #滑动平均衰减率，见《Tensorflow 实战Google深度学习框架（第一版）》p90

#定义变量
def defineVariable(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name='weights')
    biases = tf.Variable(tf.constant(0.1, shape=[shape[1]]),name='biases')

    moving_weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name='moving_weights', trainable=False)
    moving_biases = tf.Variable(tf.constant(0.1, shape=[shape[1]]),name='moving_biases', trainable=False) 
    
    #定义一个更新变量滑动平均的操作。
    update_moving_weights = moving_averages.assign_moving_average(moving_weights, weights, MOVING_AVERAGE_DECAY)  
    update_moving_biases = moving_averages.assign_moving_average(moving_biases, biases, MOVING_AVERAGE_DECAY)  
    
    #向当前计算图中添加张量集合
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_weights)       
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_biases)    
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    
    return weights, biases, moving_weights, moving_biases

#前向传播过程
def defineModel(input_tensor, global_step):
    with tf.variable_scope("layer1"):
        weights, biases, moving_weights, moving_biases = defineVariable([INPUT_NODE, LAYER1_NODE])        
        predict_layer = tf.nn.relu(tf.matmul(input_tensor, moving_weights) + moving_biases)        
        train_layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    with tf.variable_scope("layer2"):
        weights, biases, moving_weights, moving_biases = defineVariable([LAYER1_NODE, LAYER2_NODE])        
        predict_layer = tf.nn.bias_add(tf.matmul(predict_layer, moving_weights), moving_biases, name="output")        
        train_layer = tf.nn.relu(tf.matmul(train_layer, weights) + biases)
        
    with tf.variable_scope("output"):
        weights, biases, moving_weights, moving_biases = defineVariable([LAYER2_NODE, OUTPUT_NODE])        
        predict_layer = tf.nn.bias_add(tf.matmul(predict_layer, moving_weights), moving_biases, name="output")        
        train_layer = tf.matmul(train_layer, weights) + biases
        
    return predict_layer, train_layer

def train(jrain_data,train_label,validation_data,validation_label,test_data,test_label):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')  
    net = tf.reshape(x, [-1, 1, 1, INPUT_NODE], name="input")
    net = tf.reshape(net, [-1, INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    print(x)
    print(net)
    #定义训练轮数 
    #定义存储训练轮数的变量global_step。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量(trainable=Fasle)。
    #在使用TensorFlow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False,name='global_step')    
    average_y, y = defineModel(net, global_step)   
    
    #计算交叉熵及其平均值
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了 TensorFlow 中提
    #供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类问题只有一个正确答案时，即每給输入只属于一个类别，
    #可以使用这个函数来加速交叉熵的计算。这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，即logits=y
    #第二个是训练数据的非one-hot编码的类别编号，从0开始。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)                         #计算在当前 batch中所有样例的交叉熵平均值。
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    weights_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularaztion = tf.contrib.layers.apply_regularization(regularizer, weights_list)  #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
    loss = cross_entropy_mean + regularaztion                                   #总损失等于交叉熵损失和正则化损失的和。

    #设置指数衰减的学习率
    #见《Tensorflow 实战Google深度学习框架（第一版）》p85，
    #这里的LEARNING_RATE_BASE是书中的learning_rate，LEARNING_RATE_DECAY是书中的decay_rate
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                                                    #基础的学习率，随着送代的进行，更新变量时使用的学习率在这个基础上递减。
        global_step,                                                           #当前途代的轮数
        train_data.shape[0]/BATCH_SIZE,                                        #过完所有的训练数据需要的法代次数。
        LEARNING_RATE_DECAY,                                                   #学习率衰减速度 
        staircase=True)
    
    #更新参数滑动平均值的操作和通过反向传播更新变量的操作需要同时进行
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))   #tf.equal判断两个张量的每一维是否相等，如果相等返回 True,否则返回 False，correct_prediction为布尔型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))         #先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率。
    
    #初始化会话并开始训练过程。
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()                                #初始换所有变量
        validate_feed = {x: validation_data, y_: validation_label}             #准备验证数据
        test_feed = {x: test_data, y_: test_label}                             #准备测试数据
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            
            #产生这一轮使用的一个batch的训练数据,并运行训练过程。
            start= (i*BATCH_SIZE) % train_data.shape[0]
            end = min(start+BATCH_SIZE,train_data.shape[0])            
            xs=train_data[start:end,:]
            ys=train_label[start:end,:]            
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            
            #每1000轮输出1次在验证数据集上的测试结果，并保存模型
            if i % 1000 == 0:
                total_loss=sess.run(loss,feed_dict={x:xs,y_:ys})
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy is %g ,loss is %g " % (i, validate_acc,total_loss))  
                saver.save(sess,"./path/to/model/model.ckpt")

        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率，并保存模型
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))      
        saver.save(sess,"./path/to/model/model.ckpt")

        writer = tf.summary.FileWriter("./path/to/log", tf.get_default_graph())
        writer.close()
        
def main(argv=None):
    train(train_data,train_label,validation_data,validation_label,test_data,test_label)

if __name__=='__main__':
    main()
    