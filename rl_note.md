# 策略网络

策略的**输入**是状态𝑠，**输出**为动作𝑎或动作的分布![截屏2022-09-14 14.15.43](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-14 14.15.43.png)，如果是动作的分布，那么满足概率之和

![截屏2022-09-14 14.16.33](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-14 14.16.33.png)

其中𝐴为所有动作的集合。该网络表示策略，称为策略网络。将策略 函数具体化为输入节点为 4，中间多个全连接隐藏层，输出层的输出节点数为 2 的神经网 络。在交互时，选择概率最大的动作。所以｜号就是**输入｜输出**的意思

![截屏2022-09-14 14.19.00](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-14 14.19.00.png)

最简单的策略网络：

![截屏2022-09-14 14.19.31](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-14 14.19.31.png)

# PPO网络

## 重要性采样

利用重要性采样，可以使用同一个运行轨迹的记录去训练多次网络。每训练一次网络，策略网络就已经发生了变化，按理说就不能使用原先的运行轨迹了。但是又了重要性采样，只要用新概率除旧概率就行

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 21.27.04.png" alt="截屏2022-09-15 21.27.04" style="zoom:50%;" />

但是重要性采样的前提是，旧的策略网络和新的策略网络的分布不能相差太大，于是就加了一些约束

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 21.28.57.png" alt="截屏2022-09-15 21.28.57" style="zoom:50%;" />

真实环境中的奖励𝑟 并不是分布在 0 周围，很多游戏的奖励全是正数，使得𝑅(𝜏)总是 大于 0，**网络会倾向于增加所有采样到的动作的概率，而未采样到的动作出现的概率也就 相对下降**。这并不是我们希望看到的，我们希望𝑅(𝜏)能够分布在 0 周围，因此我们引入一 个偏置变量𝑏，称之为基准线，它代表了回报𝑅(𝜏)的平均水平

# 值函数方法

## 状态值函数(State Value Function，V 函数)

它定义为**从状态𝑠𝑡开始**，在策略𝜋控制下能获得的期望回报值

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 21.55.33.png" alt="截屏2022-09-15 21.55.33" style="zoom:50%;" />

状态值函数的数值**反映了当前策略下状态的好坏**，𝑉 𝜋 (𝑠𝑡 )越大，说明当前状态的总回报期望越大。

状态值函数的贝尓曼方程：

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 21.56.37.png" alt="截屏2022-09-15 21.56.37" style="zoom:50%;" />

在所有策略中，最优策略𝜋∗是指能取得𝑉𝜋(𝑠)最大值的策略，对于最优策略，同样满足贝尔曼方程

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 21.58.37.png" alt="截屏2022-09-15 21.58.37" style="zoom:50%;" />

## 状态**-**动作值函数(State-Action Value Function，Q 函数)

它定义为从状态𝑠𝑡并执行动作 𝑎𝑡的双重设定下，在策略𝜋控制下能获得的期望回报值

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 22.00.19.png" alt="截屏2022-09-15 22.00.19" style="zoom:50%;" />

Q函数和V函数的关系：

<img src="/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-15 22.03.07.png" alt="截屏2022-09-15 22.03.07" style="zoom:50%;" />

当V的下一个动作采样子V的策略时，两个期望值就想等了，此时

![截屏2022-09-17 13.30.51](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.30.51.png)

同时

![截屏2022-09-17 13.31.28](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.31.28.png)

把Q函数和V函数之间的差值定义为优势值函数，反映了在状态s下采取动作a比平均水平的差异

![截屏2022-09-17 13.32.55](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.32.55.png)

TD时差分析法

![截屏2022-09-17 13.33.44](/Users/kevin/Library/Application Support/typora-user-images/截屏2022-09-17 13.33.44.png)

这样直接就可以利用神经网络去估计t+1的Q和当前的Q

# DQN算法

利用TD时差分析法直接更新策略网络，把策略网络的损失值定义为优势值函数刚刚好

![截屏2022-09-17 13.36.47](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.36.47.png)

由于两个Q都来自于同一个网络，具有强相关性，两项措施解决。添加经验回放池，创建影子网络。影子网络的更新速度慢于训练的target网络，在代码中暂定20个epoch后，影子网络拉取target网络的新参数

![截屏2022-09-17 13.39.12](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.39.12.png)

![截屏2022-09-17 13.39.46](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.39.46.png)

![截屏2022-09-17 13.40.01](/Users/kevin/Library/Application Support/typora-user-images/截屏2022-09-17 13.40.01.png)

# Actor-Critic 方法

在Actor-Critic中存在两个网络，一个网络用来训练优势值，一个网络用来更新策略。

Actor作为策略网络，loss使用刚开始的方式进行更新

![截屏2022-09-17 13.48.32](/Users/kevin/Documents/MyCode/tensorflow_practice/rl_prac/img/截屏2022-09-17 13.48.32.png)

后面的一项就作为优势值，式子可以写成：

![截屏2022-09-17 13.49.16](/Users/kevin/Library/Application Support/typora-user-images/截屏2022-09-17 13.49.16.png)

使用优势值的时候要断开critic网络的梯度连接，同时loss添加Entropy Bonus，保证动作的概率不会太集中到某一个动作