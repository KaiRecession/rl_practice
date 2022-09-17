# gym环境尝试
# 先安装环境pip install gym
# 安装游戏库
# git clone https://github.com/openai/gym.git #拉取源代码
# cd gym # 进入目录
# pip install -e '.[all]' # 安装 Gym

import gym

# 创建平衡杆游戏环境
env = gym.make("CartPole-v1")
# 复位游戏，回到初始状态
observation = env.reset()
# 初始状态所有观测直都从[-0.05,0.05]中随机取值。
print(observation)
# 循环交互 1000 次
for _ in range(1000):
    # 显示当前时间戳的游戏画面
    env.render()
    # 随机生成一个动作
    action = env.action_space.sample()
    # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
    # 每一步都给出1的奖励，包括终止状态
    # 动作空间是离散空间: 0: 表示小车向左移动 1: 表示小车向右移动
    # 达到下列条件之一片段结束: 1、杆子与竖直方向角度超过12度 2、小车位置距离中心超过2.4（小车中心超出画面）3、片段长度超过200 4、连续100次尝试的平均奖励大于等于195。
    observation, reward, done, info = env.step(action)
    # observation = [小车位置、小车速度、杆子与竖直方向夹角、杆子的角度变化率]
    print(observation, reward, done, info)
    # 游戏回合结束，复位状态
    if done:
        observation = env.reset()

# 销毁游戏环境
env.close()
