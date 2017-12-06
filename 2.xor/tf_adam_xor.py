import datetime
import tensorflow as tf
print(datetime.datetime.now())
learning_rate = 0.01
x_data = [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]
"""
tf.placeholder(dtype, shape=None, name=None)
此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
参数：
dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
name：名称。

tf.Variable：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
声明时，必须提供初始值；
名称的真实含义，在于变量，也即在真实训练时，其值是会改变的，自然事先需要指定初始值； 
"""

x = tf.placeholder("float", shape = [None,2])
y_data = [0,1,0,1]
y = tf.placeholder("float",shape=[None,1])
weights = {
    'w1':tf.Variable(tf.random_normal([2,16])),
    'w2':tf.Variable(tf.random_normal([16,1]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([1])),
    'b2':tf.Variable(tf.random_normal([1]))
} 
def dnn(_X,_weights,_biases):
    d1 = tf.matmul(_X, _weights['w1'])+_biases['b1']
    d1 = tf.nn.relu(d1)
    d2 = tf.matmul(d1,_weights['w2'])+_biases['b2']
    d2 = tf.nn.sigmoid(d2)
    return d2
pred = dnn(x, weights, biases)

#tensorflow中有一类在tensor的某一维度上求值的函数。如：
#求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
#求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
#参数1--input_tensor:待求值的tensor。
#参数2--reduction_indices:在哪一维上求解。
#参数（3）（4）可忽略

cost = tf.reduce_mean(tf.square(y-pred))
## 优化器 http://blog.csdn.net/xierhacker/article/details/53174558
"""
Optimizer 
GradientDescentOptimizer 
AdagradOptimizer 
AdagradDAOptimizer 
MomentumOptimizer 
AdamOptimizer 
FtrlOptimizer 
RMSPropOptimizer
"""
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
"""
tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，
              如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的

tf.argmax(input, axis=None, name=None, dimension=None)

此函数是对矩阵按行或列计算最大值

参数
input：输入Tensor
axis：0表示按列，1表示按行
name：名称
dimension：和axis功能一样，默认axis取值优先。新加的字段
返回：Tensor  一般是行或列的最大值下标向量
"""
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    for _ in range(1500):
        batch_xs = tf.reshape(x_data,shape=[-1,2])
        batch_ys = tf.reshape(y_data,shape=[-1,1])
        #print(batch_xs)
        #print(batch_ys)
        sess.run(optimizer,feed_dict={x:sess.run(batch_xs),y:sess.run(batch_ys)})
        acc = sess.run(accuracy,feed_dict={x:sess.run(batch_xs),y:sess.run(batch_ys)})
        loss = sess.run(cost,feed_dict = {x:sess.run(batch_xs),y:sess.run(batch_ys)})
        #print("Step "+str(step)+",Minibatch Loss = "+"{:.6f}".format(loss)+", Training Accuracy = "+"{:.5f}".format(acc))
        step += 1
        if(step%100==0):
            print("Step "+str(step)+"    loss "+"{:.6f}".format(loss))
            print(sess.run(pred,feed_dict={x:sess.run(batch_xs)}))
        #    print(sess.run(weights))
        #    print(sess.run(biases))
print(datetime.datetime.now())
print("Optimization Finished!")
