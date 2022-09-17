from abc import ABC
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, optimizers, Model, losses
import matplotlib


matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False


env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(2222)
tf.random.set_seed(2222)
np.random.seed(2222)
# 相当于定义一个bean类
# 解除步数限制
env = gym.make('CartPole-v0').unwrapped
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        # 策略网络，也叫Actor网络，输出为概率分布pi(a|s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')

    # 添加*args, **kwargs防止出现ide提示
    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = tf.nn.softmax(x, axis=1)  # 转换成概率
        return x


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        # 偏置b的估值网络，也叫Critic网络，输出为v(s)
        self.fc1 = layers.Dense(100, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(1, kernel_initializer='he_normal')

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        # 输出基准线 b 的估计
        x = self.fc2(x)
        return x


class PPO():
    def __init__(self, gamma, batch_size, epsilon):
        super(PPO, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = []
        self.actor_optimizer = optimizers.Adam(1e-3)
        self.critic_optimizer = optimizers.Adam(3e-3)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon

    def select_action(self, s):
        s = tf.constant(s, dtype=tf.float32)
        # 不扩维度就进不去网络中，网络有一个默认batch的维度
        s =tf.expand_dims(s, axis=0)
        prob = self.actor(s)
        a = tf.random.categorical(tf.math.log(prob), 1)[0]
        a = int(a)
        # 返回动作和该动作的概率
        return a, float(prob[0][a])

    def get_value(self, s):
        s = tf.constant(s, dtype=tf.float32)
        s = tf.expand_dims(s, axis=0)
        v = self.critic(s)[0]
        # 返回的是一个标量
        return float(v)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def optimize(self):
        # 就是一个for循环取出
        state = tf.constant([t.state for t in self.buffer], dtype=tf.float32)
        action = tf.constant([t.action for t in self.buffer], dtype=tf.int32)
        action = tf.reshape(action, [-1, 1])
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = tf.constant([t.a_log_prob for t in self.buffer], dtype=tf.float32)
        old_action_log_prob = tf.reshape(old_action_log_prob, [-1, 1])
        R = 0
        Rs = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Rs.insert(0, R)
        Rs = tf.constant(Rs, dtype=tf.float32)
        # PPO利用重要性采样，可以利用相同的经验池对一个网络重复train多次
        for _ in range(round(10 * len(self.buffer) / self.batch_size)):
            # np.arange生成从0-lenth-1的列表，当作index
            # numpy.random.choice(a, size=None, replace=True, p=None)
            # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
            # replace:True表示可以取相同数字，False表示不可以取相同数字
            # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
            index = np.random.choice(np.arange(len(self.buffer)), self.batch_size, replace=False)
            # 这样写能进行两次导计算
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # 获取真实累计reward
                # tf.gather根据index的索引进行切片
                v_target = tf.expand_dims(tf.gather(Rs, index, axis=0), axis=1)
                # shape(32, 1)
                # print('v_target.shpae: ', v_target)
                # 得到critic的输出
                v = self.critic(tf.gather(state, index, axis=0))
                delta = v_target - v
                # actor的网络不会影响critic
                advantage = tf.stop_gradient(delta)

                a = tf.gather(action, index, axis=0)
                # shape(32, 1)
                # print('a.shape():', a)
                # pi.shape = (32, 2)
                pi = self.actor(tf.gather(state, index, axis=0))
                # print(pi)
                indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
                # shape=(32, 2)
                indices = tf.concat([indices, a], axis=1)
                # print('indices.shape():', indices)
                # 这是新模型的概率值
                pi_a = tf.gather_nd(pi, indices)  # 动作的概率值pi(at|st), [b]
                pi_a = tf.expand_dims(pi_a, axis=1)  # [b]=> [b,1]

                ratio = (pi_a / tf.gather(old_action_log_prob, index, axis=0))
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                # PPO误差函数
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                # 对于偏置v来说，希望与MC估计的R(st)越接近越好
                value_loss = losses.MSE(v_target, v)
                # 优化策略网络
            # tape1和tape2谁先无所谓
            grads = tape2.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            # 优化偏置值网络
            grads = tape1.gradient(value_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        self.buffer = []


def train(agent, batch_size, epoch, returns, total, print_interval):
    state = env.reset()
    for i in range(500):
        action, action_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        trans = Transition(state, action, action_prob, reward, next_state)
        agent.store_transition(trans)
        state = next_state
        total += reward
        if done:
            if len(agent.buffer) >= batch_size:
                # print(total)
                agent.optimize()
            break
    return total


def main():
    gamma = 0.98  # 激励衰减因子
    epsilon = 0.2  # PPO误差超参数0.8~1.2
    batch_size = 32  # batch size
    epoch_num = 500
    print_interval = 20

    agent = PPO(gamma, batch_size, epsilon)
    # 统计总回报
    returns = [0]
    total = 0  # 一段时间内平均回报
    for epoch in range(1, epoch_num + 1):  # 训练回合数
        total = train(agent, batch_size, epoch, returns, total, print_interval)
        if epoch % print_interval == 0:
            returns.append(total / print_interval)
            print(f"# of episode :{epoch}, avg score : {total / print_interval}")
            total = 0
    print(np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns))
    plt.plot(np.arange(len(returns)) * 20, np.array(returns), 's')
    plt.xlabel('epoch')
    plt.ylabel('avg score of every 20 epoch')
    plt.savefig('ppo_of_demo05.svg')


if __name__ == '__main__':
    main()
    print("end")
