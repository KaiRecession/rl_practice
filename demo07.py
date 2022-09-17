
import collections
from collections import namedtuple

import gym
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, optimizers, Model, losses
import matplotlib


env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(1234)
tf.random.set_seed(1234)
np.random.seed(1234)


class Qnet(Model):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = layers.Dense(256, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(256, kernel_initializer='he_normal', activation='relu')
        self.fc3 = layers.Dense(2, kernel_initializer='he_normal')

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def sample_action(self, s, epsilon):
        s = tf.constant(s, dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        # 应该是调用了自己的call方法
        out = self(s)[0]
        # 方法返回一个随机数，其在0至1的范围之内
        coin = random.random()
        # 策略改进：e - 贪心方式
        if coin < epsilon:
            # 函数返回参数1和参数2之间的任意整数， 闭区间
            return random.randint(0, 1)
        else:
            return int(tf.argmax(out))


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, trasition):
        self.buffer.append(trasition)

    def sample(self, n):
        # 从序列buffer中选择n个随机且独立的元素
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return tf.constant(s_lst, dtype=tf.float32), tf.constant(a_lst, dtype=tf.int32), \
               tf.constant(r_lst, dtype=tf.float32), tf.constant(s_prime_lst, dtype=tf.float32), \
               tf.constant(done_mask_lst, dtype=tf.float32)

    def size(self):
        return len(self.buffer)


def train(q, q_target, memory, optimizer, batch_size, gamma):
    huber = losses.Huber()
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        with tf.GradientTape() as tape:
            q_out = q(s)
            indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
            # 为了方便挑选a的概率
            indices = tf.concat([indices, a], axis=1)
            q_a = tf.gather_nd(q_out, indices)
            q_a = tf.expand_dims(q_a, axis=1)
            # 来自影子网络的Q
            # 用下一个状态放入影子网络求的下一个状态开始的最大Q值，但是不训练影子网络
            max_q_prime = tf.reduce_max(q_target(s_prime), axis=1, keepdims=True)
            target = r + gamma * max_q_prime * done_mask

            loss = huber(q_a, target)

        grads = tape.gradient(loss, q.trainable_variables)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))


def main():
    learning_rate = 0.0002
    gamma = 0.99
    buffer_limit = 50000
    batch_size = 32
    epoch_num = 10000

    env = gym.make('CartPole-v1')
    q_net = Qnet()
    q_target = Qnet()
    q_net.build(input_shape=(2, 4))
    q_target.build(input_shape=(2, 4))
    for src, dest in zip(q_net.variables, q_target.variables):
        # 将src的变量付给dest
        dest.assign(src)
    memory = ReplayBuffer(buffer_limit)
    print_interval = 20
    score = 0.0
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epoch_num):
        # epsilon概率也会8%到1%衰减，越到后面越使用Q值最大的动作
        epsilon = max(0.01, 0.08 - 0.01 * (epoch / 200))
        s = env.reset()
        for t in range(600):
            # 还是让网络输出一个动作
            a = q_net.sample_action(s, epsilon)
            # s_prime = next_state
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break
        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if epoch % print_interval == 0 and epoch != 0:
            for src, dest in zip(q_net.variables, q_target.variables):
                dest.assign(src)
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, " \
                  "epsilon : {:.1f}%" \
                  .format(epoch, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()

