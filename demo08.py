import multiprocessing
import threading

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from queue import Queue
from tensorflow.keras import layers, optimizers, Model


plt.rcParams['font.size'] = 18
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.figsize'] = [9, 7]
plt.rcParams['font.family'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

tf.random.set_seed(1231)
np.random.seed(1231)

class ActorCritic(Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(action_size)

        self.fc3 = layers.Dense(128, activation='relu')
        self.fc4 = layers.Dense(1)

    def call(self, inputs):
        # 还是定义了两个小网络，上面是actor，下面是critic
        x = self.fc1(inputs)
        logits = self.fc2(x)

        v = self.fc3(inputs)
        values = self.fc4(v)

        return logits, values


def record(epoch, epoch_reward, worker_id, global_epoch_reward, result_queue, total_loss, num_steps):
    if global_epoch_reward == 0:
        global_epoch_reward = epoch_reward
    else:
        global_epoch_reward = global_epoch_reward * 0.99 + epoch_reward * 0.01
    print(
        f"{epoch} | "
        f"Average Reward: {int(global_epoch_reward)} | "
        f"Episode Reward: {int(epoch_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_id}"
    )
    result_queue.put(global_epoch_reward)  # 保存回报，传给主线程
    return global_epoch_reward


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Agent:
    def __init__(self):
        self.opt = optimizers.Adam(1e-3)
        self.server = ActorCritic(4, 2)
        self.server(tf.random.normal((2, 4)))

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.server, self.opt, res_queue, i)
                   for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print("starting worker{}".format(i))
            worker.start()
        returns = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                returns.append(reward)
            else:
                break
        [w.join() for w in workers]

        print(returns)
        plt.figure()
        plt.plot(np.arange(len(returns)), returns)
        plt.xlabel('epochs')
        plt.ylabel('scores')
        plt.savefig('demo08.svg')


class Worker(threading.Thread):
    def __init__(self, server, opt, result_queue, id):
        super(Worker, self).__init__()
        self.result_queue = result_queue
        self.server = server
        self.opt = opt
        self.client = ActorCritic(4, 2)
        self.worker_id = id
        self.env = gym.make('CartPole-v1').unwrapped
        self.ep_loss = 0.0

    def run(self):
        # 相当于存储一整条轨迹的类
        mem = Memory()
        # 每个worker线程的epoch次数
        for epi_counter in range(500):
            current_state = self.env.reset()
            mem.clear()
            epoch_reward = 0.
            epoch_steps = 0
            done = False
            while not done:
                # 变换维度，方便网络输入
                logits, _ = self.client(tf.constant(current_state[None, :], dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                # 按照概率选择action
                action = np.random.choice(2, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                # 相当于word里面的一次轨迹的reward总和，就是为了方便展示信息
                epoch_reward += reward
                mem.store(current_state, action, reward)
                # 这个step是本次轨迹走过的步数
                epoch_steps += 1
                current_state = new_state

                if epoch_steps >= 500 or done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem)
                    # worker线程的梯度拿出来
                    grads = tape.gradient(total_loss, self.client.trainable_variables)
                    # 把worker线程的梯度给server更新参数
                    self.opt.apply_gradients(zip(grads, self.server.trainable_variables))
                    # 把server参数更新到client
                    self.client.set_weights(self.server.get_weights())
                    mem.clear()
                    self.result_queue.put(epoch_reward)
                    print("epoch=%s," % epi_counter, "worker=%s," % self.worker_id, "reward=%s" % epoch_reward)
                    break

        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        # 如果进来的state不是因为done，那么最后一步的之后的期望reward就用网络计算
        if done:
            reward_sum = 0.
        else:
            # 把网络输出的value变成一个普通常量
            reward_sum = self.client(tf.constant(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        # 倒着计算reward后，把列表反转，回到正常的顺序
        discounted_rewards.reverse()
        # 此时得到的概率，值都是批次所得
        logits, values = self.client(tf.constant(np.vstack(memory.states), dtype=tf.float32))
        # 计算advantage = R() - v(s)
        advantage = tf.constant(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        value_loss = advantage ** 2
        policy = tf.nn.softmax(logits)
        # 计算多分类的交叉熵
        # 就是一步计算了-log对应的概率，直接让loss函数连负号都不用加
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
        policy_loss = policy_loss * tf.stop_gradient(advantage)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        # 保留一些试探性的选择，输出选项的各个概率值之间差距越大，entropy就越小，适当增大entropy，保留一些试探性的选择
        policy_loss = policy_loss - 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == '__main__':
    master = Agent()
    master.train()
