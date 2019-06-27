
# Section 0 问题描述与完成项目流程

## 1. 问题描述


![](https://img-blog.csdnimg.cn/2019062721473076.png#pic_center =160x)

在该项目中，你将使用强化学习算法，实现一个自动走迷宫机器人。

1. 如上图所示，智能机器人显示在右上角。在我们的迷宫中，有陷阱（红色炸弹）及终点（蓝色的目标点）两种情景。机器人要尽量避开陷阱、尽快到达目的地。
2. 小车可执行的动作包括：向上走 `u`、向右走 `r`、向下走 `d`、向左走 `l`。
3. 执行不同的动作后，根据不同的情况会获得不同的奖励，具体而言，有以下几种情况。
    - 撞到墙壁：-10
    - 走到终点：50
    - 走到陷阱：-30
    - 其余情况：-0.1
4. 我们需要通过修改 `robot.py` 中的代码，来实现一个 Q Learning 机器人，实现上述的目标。

## 2. 完成项目流程

1. 配置环境，使用 `envirnment.yml` 文件配置名为 `robot-env` 的 conda 环境，具体而言，你只需转到当前的目录，在命令行/终端中运行如下代码，稍作等待即可。
```
conda env create -f envirnment.yml
```
安装完毕后，在命令行/终端中运行 `source activate robot-env`（Mac/Linux 系统）或 `activate robot-env`（Windows 系统）激活该环境。

2. 阅读 `main.ipynb` 中的指导完成项目，并根据指导修改对应的代码，生成、观察结果。
3. 导出代码与报告，上传文件，提交审阅并优化。
---

---

# Section 1 算法理解


## 1. 1 强化学习总览

强化学习作为机器学习算法的一种，其模式也是让智能体在“训练”中学到“经验”，以实现给定的任务。但不同于监督学习与非监督学习，在强化学习的框架中，我们更侧重通过智能体与环境的**交互**来学习。通常在监督学习和非监督学习任务中，智能体往往需要通过给定的训练集，辅之以既定的训练目标（如最小化损失函数），通过给定的学习算法来实现这一目标。然而在强化学习中，智能体则是通过其与环境交互得到的奖励进行学习。这个环境可以是虚拟的（如虚拟的迷宫），也可以是真实的（自动驾驶汽车在真实道路上收集数据）。


在强化学习中有五个核心组成部分，它们分别是：**环境（Environment）**、**智能体（Agent）**、**状态（State）**、**动作（Action）**和**奖励（Reward）**。在某一时间节点 $t$：
    
- 智能体在从环境中感知其所处的状态 $s_t$
- 智能体根据某些准则选择动作 $a_t$
- 环境根据智能体选择的动作，向智能体反馈奖励 $r_{t+1}$

通过合理的学习算法，智能体将在这样的问题设置下，成功学到一个在状态 $s_t$ 选择动作 $a_t$ 的策略 $\pi (s_t) = a_t$。

---

**问题 1**：请参照如上的定义，描述出 “机器人走迷宫这个问题” 中强化学习五个组成部分对应的实际对象：

- **环境** : 环境指整个迷宫，包括陷阱和目标点；
- **状态** : 状态指的是机器人所处的位置；


- **动作** : 包括向上走 u、向右走 r、向下走 d、向左走 l。
- **奖励** : 撞到墙壁：-10；走到终点：50；走到陷阱：-30；其余情况：-0.1

$$
T(s^{'}, a, s) = P(s^{'}|a,s)
$$

---

## 1.2 计算 Q 值 

在我们的项目中，我们要实现基于 Q-Learning 的强化学习算法。Q-Learning 是一个值迭代（Value Iteration）算法。与策略迭代（Policy Iteration）算法不同，值迭代算法会计算每个”状态“或是”状态-动作“的值（Value）或是效用（Utility），然后在执行动作的时候，会设法最大化这个值。因此，对每个状态值的准确估计，是我们值迭代算法的核心。通常我们会考虑**最大化动作的长期奖励**，即不仅考虑当前动作带来的奖励，还会考虑动作长远的奖励。

在 Q-Learning 算法中，我们把这个长期奖励记为 Q 值，我们会考虑每个 ”状态-动作“ 的 Q 值，具体而言，它的计算公式为：

$$
q(s_{t},a) = R_{t+1} + \gamma \times\max_a q(a,s_{t+1})
$$

也就是对于当前的“状态-动作” $(s_{t},a)$，我们考虑执行动作 $a$ 后环境给我们的奖励 $R_{t+1}$，以及执行动作 $a$ 到达 $s_{t+1}$后，执行任意动作能够获得的最大的Q值 $\max_a q(a,s_{t+1})$，$\gamma$ 为折扣因子。

不过一般地，我们使用更为保守地更新 Q 表的方法，即引入松弛变量 $alpha$，按如下的公式进行更新，使得 Q 表的迭代变化更为平缓。

$$
q(s_{t},a) = (1-\alpha) \times q(s_{t},a) + \alpha \times(R_{t+1} + \gamma \times\max_a q(a,s_{t+1}))
$$



---
![ALT](https://img-blog.csdnimg.cn/20190627214934358.png#pic_center =160x)

**问题 2**：根据已知条件求 $q(s_{t},a)$，在如下模板代码中的空格填入对应的数字即可。


已知：如上图，机器人位于 $s_1$，行动为 `u`，行动获得的奖励与题目的默认设置相同。在 $s_2$ 中执行各动作的 Q 值为：`u`: -24，`r`: -13，`d`: -0.29、`l`: +40，$\gamma$ 取0.9。
$$
q(s_{t},a)  = R_{t+1} + \gamma \times\max_a q(a,s_{t+1})  =(-0.1) + (0.9)*(40)  =(35.9) 
$$


---

## 1.3 如何选择动作

在强化学习中，「探索-利用」问题是非常重要的问题。具体来说，根据上面的定义，我们会尽可能地让机器人在每次选择最优的决策，来最大化长期奖励。但是这样做有如下的弊端：
1. 在初步的学习中，我们的 Q 值会不准确，如果在这个时候都按照 Q 值来选择，那么会造成错误。
2. 学习一段时间后，机器人的路线会相对固定，则机器人无法对环境进行有效的探索。

因此我们需要一种办法，来解决如上的问题，增加机器人的探索。由此我们考虑使用 epsilon-greedy 算法，即在小车选择动作的时候，以一部分的概率随机选择动作，以一部分的概率按照最优的 Q 值选择动作。同时，这个选择随机动作的概率应当随着训练的过程逐步减小。

---

**问题 3**：在如下的代码块中，实现 epsilon-greedy 算法的逻辑，并运行测试代码。


```python
import numpy as np
import random

actions = ['u','r','d','l']
qline = {'u':1.2, 'r':-2.1, 'd':-24.5, 'l':27}
epsilon = 0.3 # 以0.3的概率进行随机选择

def choose_action(epsilon=None):
    
    action = None
    if random.uniform(0,1) < epsilon: # 以某一概率
        action = random.choice(actions) # 随机选择
    else: 
        action = max(qline, key=qline.get) # 否则选择具有最大 Q 值的动作
    
    return action
```


```python
action = choose_action(epsilon)
print("以epsilon的概率随机选择动作：", action)
```

    以epsilon的概率随机选择动作： d
    

## 1.4 如何选择强化学习参数

* **epsilon：**

**Q: 如何理解 greed-epsilon 方法／如何设置 epsilon／如何理解 exploration & exploitation 权衡？**

A: (1) 我们的小车一开始接触到的 state 很少，并且如果小车按照已经学到的 qtable 执行，那么小车很有可能出错或者绕圈圈。同时我们希望小车一开始能随机的走一走，接触到更多的 state。(2) 基于上述原因，我们希望小车在一开始的时候不完全按照 Q learning 的结果运行，即以一定的概率 epsilon，随机选择 action，而不是根据 maxQ 来选择 action。然后随着不断的学习，那么我会降低这个随机的概率，使用一个衰减函数来降低 epsilon。(3) 这个就解决了所谓的 exploration and exploitation 的问题，在“探索”和“执行”之间寻找一个权衡。

* **alpha**

alpha 是一个权衡上一次学到结果和这一次学习结果的量，如：$$Q = (1-alpha)*Q_old + alpha*Q_current$$
alpha 设置过低会导致机器人只在乎之前的知识，而不能积累新的 reward。一般取 0.5 来均衡以前知识及新的 reward。

* **gamma**

gamma 是考虑未来奖励的因子，是一个(0,1)之间的值。一般我们取0.9，能够充分地对未来奖励（也就是执行了这一步动作之后，在对应的下一状态下，执行动作会带来什么奖励）进行考虑。

实际上如果你将它调小了，你会发现终点处的正奖励不能够“扩散”到周围，也就是说，机器人很有可能无法学习到一个到达终点的策略。你可以自己尝试一下。



---

---

# Section 2 代码实现

## 2.1. `Maze` 类理解

我们首先引入了迷宫类 `Maze`，这是一个非常强大的函数，它能够根据你的要求随机创建一个迷宫，或者根据指定的文件，读入一个迷宫地图信息。

1. 使用 `Maze("file_name")` 根据指定文件创建迷宫，或者使用 `Maze(maze_size=(height,width))` 来随机生成一个迷宫。
2. 使用 `trap_number` 参数，在创建迷宫的时候，设定迷宫中陷阱的数量。
3. 直接键入迷宫变量的名字按回车，展示迷宫图像（如 `g=Maze("xx.txt")`，那么直接输入 `g` 即可。
4. 建议生成的迷宫尺寸，长在 6~12 之间，宽在 10～12 之间。

---

**问题 4**：在如下的代码块中，创建你的迷宫并展示。


```python
from Maze import Maze
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

## todo: 创建迷宫并展示
# first_maze = Maze(maze_size=(10, 12), trap_number=5)
first_maze = Maze(from_file='test_world/maze_01.txt')
first_maze
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215424430.png#pic_center =320x)





    Maze of size (12, 12)



---

你可能已经注意到，在迷宫中我们已经默认放置了一个机器人。实际上，我们为迷宫配置了相应的 API，来帮助机器人的移动与感知。其中你随后会使用的两个 API 为 `maze.sense_robot()` 及 `maze.move_robot()`。

1. `maze.sense_robot()` 为一个无参数的函数，输出机器人在迷宫中目前的位置。
2. `maze.move_robot(direction)` 对输入的移动方向，移动机器人，并返回对应动作的奖励值。

---

**问题 5**：随机移动机器人，并记录下获得的奖励，展示出机器人最后的位置。


```python
import matplotlib.pyplot as plt

rewards = []

## 循环、随机移动机器人10次，记录下奖励
for i in range(10):
    direction = np.random.choice(actions)
    print("direction: ", direction)
    reward = first_maze.move_robot(direction)
    rewards.append(reward)

## 输出机器人最后的位置
print("机器人最后的位置:", first_maze.sense_robot())

## 打印迷宫，观察机器人位置
first_maze

# current_maze = first_maze.draw_current_maze()

# plt.figure(figsize=(first_maze.height, first_maze.width))
# plt.imshow(current_maze)
# plt.axis('off')
# plt.show()
```

    direction:  u
    direction:  l
    direction:  u
    direction:  u
    direction:  r
    direction:  d
    direction:  d
    direction:  l
    direction:  r
    direction:  u
    机器人最后的位置: (1, 11)
    

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215445520.png#pic_center =320x)





    Maze of size (12, 12)



## 2.2. `Robot` 类实现

`Robot` 类是我们需要重点实现的部分。在这个类中，我们需要实现诸多功能，以使得我们成功实现一个强化学习智能体。总体来说，之前我们是人为地在环境中移动了机器人，但是现在通过实现 `Robot` 这个类，机器人将会自己移动。通过实现学习函数，`Robot` 类将会学习到如何选择最优的动作，并且更新强化学习中对应的参数。

首先 `Robot` 有多个输入，其中 `alpha=0.5, gamma=0.9, epsilon0=0.5` 表征强化学习相关的各个参数的默认值，这些在之前你已经了解到，`Maze` 应为机器人所在迷宫对象。

随后观察 `Robot.update` 函数，它指明了在每次执行动作时，`Robot` 需要执行的程序。按照这些程序，各个函数的功能也就明了了。

最后你需要实现 `Robot.py` 代码中的8段代码，他们都在代码中以 `#TODO` 进行标注，你能轻松地找到他们。

---

**问题 6**：实现 `Robot.py` 中的8段代码，并运行如下代码检查效果（记得将 `maze` 变量修改为你创建迷宫的变量名）。




```python
from Robot import Robot
robot = Robot(first_maze) # 记得将 maze 变量修改为你创建迷宫的变量名
robot.set_status(learning=True,testing=False)
print(robot.update())

first_maze
```

    ('u', -0.1)
    


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215458495.png#pic_center =320x)





    Maze of size (12, 12)



---

## 2.3 用 `Runner` 类训练 Robot

在实现了上述内容之后，我们就可以开始对我们 `Robot` 进行训练并调参了。我们为你准备了又一个非常棒的类 `Runner`，来实现整个训练过程及可视化。使用如下的代码，你可以成功对机器人进行训练。并且你会在当前文件夹中生成一个名为 `filename` 的视频，记录了整个训练的过程。通过观察该视频，你能够发现训练过程中的问题，并且优化你的代码及参数。

---

**问题 7**：尝试利用下列代码训练机器人，并进行调参。可选的参数包括：

- 训练参数
    - 训练次数 `epoch`
- 机器人参数：
    - `epsilon0` (epsilon 初值)
    - `epsilon`衰减（可以是线性、指数衰减，可以调整衰减的速度），你需要在 Robot.py 中调整
    - `alpha`
    - `gamma`
- 迷宫参数:
    - 迷宫大小
    - 迷宫中陷阱的数量


```python
from Runner import Runner
from os.path import join

# g = Maze(maze_size=maze_size,trap_number=trap_number)
g = Maze(from_file='test_world/maze_01.txt')
```


```python
## 可选的参数：
epoch = 50

epsilon0 = 0.8
alpha = 0.8
gamma = 0.1

# maze_size = (6,6)
# trap_number = 1

r = Robot(g,alpha=alpha, epsilon0=epsilon0, gamma=gamma)
r.set_status(learning=True)

runner = Runner(r, g)
runner.run_training(epoch, display_direction=True)
runner.generate_movie(filename = join("videos/","robot-"+str((epsilon0, alpha, gamma, epoch))+".mp4")) # 你可以注释该行代码，加快运行速度，不过你就无法观察到视频了。

runner.plot_results()
```

    Generate Movies: 100%|█████████████████████████████████████████████████████████████| 2213/2213 [01:43<00:00, 21.39it/s]
    


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215509342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)


---

使用 `runner.plot_results()` 函数，能够打印机器人在训练过程中的一些参数信息。

- Success Times 代表机器人在训练过程中成功的累计次数，这应当是一个累积递增的图像。
- Accumulated Rewards 代表机器人在每次训练 epoch 中，获得的累积奖励的值，这应当是一个逐步递增的图像。
- Running Times per Epoch 代表在每次训练 epoch 中，小车训练的次数（到达终点就会停止该 epoch 转入下次训练），这应当是一个逐步递减的图像。

---

**问题 8**：使用 `runner.plot_results()` 输出训练结果，根据该结果对你的机器人进行分析。

- 指出你选用的参数如何，选用参数的原因。
- 建议你比较不同参数下机器人的训练的情况。
- 训练的结果是否满意，有何改进的计划。

## 2.4 实验结果

在这次实验中，我尝试了调整四种不同的参数：训练次数epoch、alpha、gamma和epsilon0，以及epsilon衰减方式。
* 随着训练次数的增大，算法逐渐收敛到最优策略，在这次的实验环境中，选择的test_world中的第一个迷宫，根据环境的复杂程度，设置为50较为合适；
* alpha用于平衡上一次学到结果和这一次学习结果，一般取 0.5 来均衡以前知识及新的 reward。
* gamma用于考虑未来奖励，一般取0.9，能够充分地对未来奖励（也就是执行了这一步动作之后，在对应的下一状态下，执行动作会带来什么奖励）进行考虑。
* epsilon用于智能体选择动作，一般而言，希望智能体在初期的时候能够探索较多的state，在学习到一定的知识后，选择最佳动作，所以我认为设一个较大的初值。
* 一般而言，可以采用线性衰减和指数衰减，线性衰减算法收敛速度较快。

以下是我的实验过程和参数选择结果：

**第一步：**

* **指数衰减，epsilon0=0.3，alpha=0.5, gamma=0.9**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019062721552744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

* **指数衰减，epsilon0=0.5，alpha=0.5, gamma=0.9**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215537717.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

* **指数衰减，epsilon0=0.8，alpha=0.5, gamma=0.9**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215633749.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

结论： **当增大epsilon初始值时，发现智能体在前期会探索更多的state，epsilon=0.3时，智能体前期比较倾向于选择Q值较大的动作，容易重复，在这次实验中，我选择初步选择epsilon0=0.8**

**第二步：指数衰减改为线性衰减**

* **线性衰减，epsilon0=0.8，alpha=0.5, gamma=0.9**
![image.png](https://img-blog.csdnimg.cn/20190627215651530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

* **线性衰减，epsilon0=0.5，alpha=0.5, gamma=0.9**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215704430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

结论：**可以发现，采用epsilon分段线性衰减比指数衰减算法收敛较快，因此我最后选用分段线性衰减， epsilon初始值为0.8，计算公式如下：**

$$epsilon=\begin{cases} epsilon\ / \  2, & \text{if epsilon<0.1} \\ epsilon0-0.1*t, & \text{if epsilon>=0.1} \end{cases}$$

**第三步：调整alpha值**

**线性衰减，epsilon0=0.8，alpha=0.1, gamma=0.9**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215719871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

**线性衰减，epsilon0=0.8，alpha=0.8, gamma=0.9**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215726913.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

结论：**当alpha值较低时，算法更多关注之前的知识，导致算法学习能力较差；提高alpha值时，算法更多注重探索得到的奖励，开始的时候震荡较多，但是但是很快能够收敛到最优策略，在这次实验中，并没有发现提高alpha带来明显的提高，alpha仍然选择0.5。**

**第三步：调节gamma**

gamma越小时，智能体越关注较近的回报，在智能体环境较复杂时，算法可能无法收敛到最优策略。

**线性衰减，epsilon0=0.8，alpha=0.1, gamma=0.1**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190627215741400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5MzE3NjE3,size_16,color_FFFFFF,t_70)

**最终参数：线性衰减，epsilon0=0.8，alpha=0.5, gamma=0.9，epoch=50**

**改进：**
* 可以考虑选择其他的学习方式，如Sarsa和Sarsamax等；
* 不断试错，选择最优参数；
* 实际场景的复杂度远远高于此次实验迷宫的复杂度，因此需要加深Reinforcement Learning的学习

---


