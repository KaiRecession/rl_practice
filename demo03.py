# demo02的自写版本
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
import gym
import numpy as np

env = gym.make('CartPole-v1')  # 创建游戏环境
env = gym.make('CartPole-v0').unwrapped
env.seed(2333)
tf.random.set_seed(2333)
np.random.seed(2333)


class Policy(Model):
    def __init__(self, learning_rate, gamma):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = layers.Dense(128, kernel_initializer='he_normal', activation='relu')
        self.fc2 = layers.Dense(2, kernel_initializer='he_normal')
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.gamma = gamma

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = x = tf.nn.softmax(self.fc2(x), axis=1)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, tape):
        R = 0
        # 倒着取
        for r, log_prob in self.data[::-1]:
            R = r + self.gamma * R
            # 期望被变形后最后是和log的概率有关的，所以loss只要写log概率就行，其他都是幅度问题，只要方向对了就行
            # 这个概率把梯度连了起来
            loss = -log_prob * R
            # stop后面再有变量的相关算式计算，不会记录梯度，在这里基本没有用
            with tape.stop_recording():
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.data = []


def train(epoch, pi, print_interval, returns, score):
    s = env.reset()
    # persistent变量，默认情况下，调用一次求导之后，GradientTape所持有的资源就会被释放，不能再执行，如果需要持续求导，persistent默认为False,，也就是g只能调用一次，如果指定persistent为true，则可以多次求导。
    # 是false的时候tape.gradient只能够调用一次，True的时候可以调用多次
    with tf.GradientTape(persistent=True) as tape:
        for t in range(501):
            s = tf.constant(s, dtype=tf.float32)
            s = tf.expand_dims(s, axis=0)
            # 动作策略网络
            prob = pi(s)
            # 从类别分布中采样1个动作, shape: [1]
            # 按照概率随机选取prob的索引，概率越大选取的索引可能越大。选取个数看后面的number。维度和prob一样，此处prob的shape为【1， 2】
            a = tf.random.categorical(tf.math.log(prob), 1)[0]
            # tensor转换为int类型，a表示着索引
            a = int(a)
            s_prime, r, done, info = env.step(a)
            pi.put_data((r, tf.math.log(prob[0][a])))
            s = s_prime
            score += r

            if epoch > 1000:
                env.render()

                # 当前episode终止
            if done:
                # print(score)
                break
        # 每次交互完毕再训练
        pi.train_net(tape)
    del tape
    return score



def main():
    learning_rate = 0.0002
    gamma = 0.98

    pi = Policy(learning_rate, gamma)  # 创建策略网络
    pi(tf.random.normal((4, 4)))
    pi.summary()
    score = 0.0  # 计分
    print_interval = 20  # 打印间隔
    returns = [0]
    epoch_num = 400

    for epoch in range(1, epoch_num + 1):
        score = train(epoch, pi, print_interval, returns, score)
        if epoch % print_interval == 0 and epoch != 0:
            # print(score)
            returns.append(score / print_interval)
            # 每20次的平均得分
            print(f"# of episode :{epoch}, avg score : {score / print_interval}")
            score = 0.0

    # 关闭环境
    env.close()

    plt.plot(np.arange(len(returns)) * print_interval, returns)
    plt.plot(np.arange(len(returns)) * print_interval, returns, 's')
    plt.xlabel('epoch')
    plt.ylabel('avg score of every 20 epoch')
    plt.savefig('policy_of_demo03.svg')


if __name__ == '__main__':
    main()
