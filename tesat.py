import numpy as np
import tensorflow as tf
a = [1, 2, 3, 5]
for i in a[::-1]:
    print(i)
print([m for m in a])
print(np.arange(10))
x = tf.Variable(3.0, trainable=True)
m = tf.Variable(2.0, trainable=True)
print(int(0.4), int(1.6))
x = tf.Variable(4.0)
prob = tf.convert_to_tensor(np.array([[0.2, 0.1, 0.7]]))
prob = tf.math.log(prob)
print(prob)
a = tf.random.categorical(prob, 20)
print('a:', a)
with tf.GradientTape(persistent=True) as tape:

    y = tf.pow(x, 2)

    z=tape.gradient(y,x)

   # The gradient computation below is not traced, saving memory.

    y+=x

    dy_dx = tape.gradient(y, x)

print(z)

print(dy_dx)