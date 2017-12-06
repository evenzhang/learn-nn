import tensorflow as tf
sess = tf.Session()
hello=tf.constant('Hello,Tensorflow!')
print(sess.run(hello))
