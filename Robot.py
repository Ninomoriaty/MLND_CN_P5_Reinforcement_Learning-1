import random
import numpy as np
 
class Robot(object):

    # 定义初始化参数
    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze 
        self.valid_actions = self.maze.valid_actions # 引用maze中的定义
        self.state = None # 状态， 位置参数， （x， y）
        self.action = None # 动作

        # Set Parameters of the Learning Robot
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣率， 即奖励衰减值

        self.epsilon0 = epsilon0 # epsilon greedy策略的参数原值
        self.epsilon = epsilon0 # epsilon 衰减（可以是线性、指数衰减，可以调整衰减的速度）
        self.t = 0 # 步数

        self.Qtable = {} # Q值表
        self.reset() # 重置

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state() # 调用内部函数sense_state，返回现在的状态值
        self.create_Qtable_line(self.state) # 更新Q表，如果没有，新增该状态下的值，注意这里的Q表是字典，不是数组

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            # self.epsilon = 0
            pass

        else:
            # TODO 2. Update parameters when learning
            self.t += 1
            # self.epsilon = self.epsilon ** self.t
            if self.epsilon < 0.01:
                self.epsilon = 0.01
            else:
                self.epsilon -= self.t * 0.1

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """
        # TODO 3. Return robot's current state
        return self.maze.sense_robot() # 要调用maze中函数获得当前位置（状态）

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if state in self.Qtable:
            pass
        else:
            self.Qtable[state] = {'u':0.0, 'r':0.0, 'd':0.0, 'l':0.0}  # 否则，新增一个状态，并赋值为0，注意是float

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return random.choice(self.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                aciton = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
                return aciton

        elif self.testing:
            # TODO 7. choose action with highest q value
            aciton = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
            return aciton

        else:

            # TODO 6. Return random choose aciton
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
        
            # TODO 8. When learning, update the q table according
            # to the given rules
            old_Q = self.Qtable[self.state][action]
            self.Qtable[self.state][action] = (1 - self.alpha) * old_Q + \
                self.alpha * (r + self.gamma * float(max(self.Qtable[next_state].values())))


    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
