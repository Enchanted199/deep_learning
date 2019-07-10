import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

coefficients = np.array([[1.], [-20, ], [25, ]])
w = tf.Variable(0, dtype=tf.float32)
# cost = w**2-10*w+25
x = tf.placeholder(tf.float32, [3, 1])
# cost = tf.add(tf.add(w**2,tf.multiply(-10,w)),25)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]
train = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.compat.v1.global_variables_initializer()
session = tf.compat.v1.Session()
session.run(init)
print(session.run(w))
session.run(train, feed_dict={x: coefficients})
for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
print(session.run(w))
