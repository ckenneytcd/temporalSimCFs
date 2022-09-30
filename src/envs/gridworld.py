import random

import gym
import numpy as np


class Gridworld(gym.Env):

    def __init__(self):
        self.world_dim = 5

        self.state_dim = self.world_dim ** 2 + 1

        self.chopping = 0
        self.max_chopping = 10

        self.step_pen = -1
        self.goal_rew = 10

        self.max_steps = 100
        self.steps = 0

        self.lows = np.array([0]*self.state_dim)
        self.highs = np.array([3] * 25 + [self.max_chopping])
        self.observation_space = gym.spaces.Box(self.lows, self.highs, shape=(26, ))
        self.action_space = gym.spaces.Discrete(6)

        self.state = np.zeros((self.state_dim, ))

        self.num_trees = 2

        self.ACTIONS = {'RIGHT': 0, 'DOWN': 1, 'LEFT': 2, 'UP': 3, 'CHOP': 4, 'SHOOT': 5}
        self.OBJECTS = {'AGENT': 1, 'MONSTER': 2, 'TREE': 3}

    def step(self, action):
        if isinstance(action, str):
            action = self.ACTIONS[action]

        new_state, done, rew = self.get_new_state(self.state, action)

        self.state = new_state
        self.steps += 1

        return new_state.flatten(), rew, done, {}

    def create_state(self, agent, monster, trees, chopping, chopped_trees=[]):
        state = [0.0] * self.state_dim
        state[-1] = chopping
        state[agent] = self.OBJECTS['AGENT']
        state[monster] = self.OBJECTS['MONSTER']
        for t in trees:
            if t not in chopped_trees:
                state[t] = self.OBJECTS['TREE']

        return np.array(state)

    def get_new_state(self, state, action):
        agent = np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['AGENT'])[0][0]
        monster = np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['MONSTER'])[0][0]
        trees = list(np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['TREE'])[0])

        facing_monster = self.facing_obstacle(agent, [monster], action)
        facing_tree = self.facing_obstacle(agent, trees, action)

        chopped_trees = []

        if action == 0:  # MOVE
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if (agent + 1) % self.world_dim != 0:
                    agent += 1
        elif action == 1:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent + self.world_dim < self.world_dim * self.world_dim:
                    agent += self.world_dim
        elif action == 2:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent % self.world_dim != 0:
                    agent -= 1
        elif action == 3:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent >= self.world_dim:
                    agent -= self.world_dim
        elif action == 4:  # CHOP
            near_trees = self.get_neighboring_trees(agent, trees)
            if len(near_trees):
                t = near_trees[0] # start with first tree
                self.chopping += 1
                if self.chopping >= self.max_chopping:
                    chopped_trees.append(t)
                    self.chopping = 0

        elif action == 5:  # SHOOT
            self.chopping = 0
            if (int(agent / self.world_dim) == int(monster / self.world_dim)) or (agent % self.world_dim == monster % self.world_dim):
                free = self.check_if_path_free(agent, monster, trees)
                if free:
                    new_array = self.create_state(agent, monster, trees, self.chopping)
                    return new_array, True, self.goal_rew

        new_state = self.create_state(agent, monster, trees, self.chopping, chopped_trees)

        self.state = new_state

        return new_state, self.steps >= self.max_steps, self.step_pen

    def get_neighboring_trees(self, agent, trees):
        nts = []
        for t in trees:
            if self.next_to_obstacle(agent, t):
                nts.append(t)

        return nts

    def facing_obstacle(self, agent, obstacles, action):
        for o in obstacles:
            if ((agent + 1 == o) and ((agent + 1) % self.world_dim != 0) and  action == self.ACTIONS['RIGHT']) \
                    or (agent + self.world_dim == o and action == self.ACTIONS['DOWN']) \
                    or ((agent - 1 == o) and (agent % self.world_dim != 0) and action == self.ACTIONS['LEFT']) \
                    or (agent - self.world_dim == o and action == self.ACTIONS['UP']):
                return True

        return False

    def next_to_obstacle(self, agent, obstacle):
        if ((agent + 1 == obstacle) and ((agent + 1) % self.world_dim != 0)) \
                or (agent + self.world_dim == obstacle) \
                or ((agent - 1 == obstacle) and (agent % self.world_dim != 0)) \
                or (agent - self.world_dim == obstacle):
            return True

        return False

    def check_if_path_free(self, agent, monster, trees):
        if int(agent / self.world_dim) == int(monster / self.world_dim):
            for t in trees:
                if t > min([agent, monster]) and t < max([agent, monster]):
                    return False

        if (agent % self.world_dim == monster % self.world_dim):
            for t in trees:
                if t % self.world_dim == monster % self.world_dim and t > min([agent, monster]) and t < max([agent, monster]):
                    return False

        return True

    def reset(self):
        monster = random.randint(0, self.world_dim * self.world_dim - 1)
        agent = random.randint(0, self.world_dim * self.world_dim - 1)

        while agent % 5 > 1:
            agent = random.randint(0, self.world_dim * self.world_dim - 1)

        while monster % 5 < 3:
            monster = random.randint(0, self.world_dim * self.world_dim - 1)

        tree_wall = [2, 7, 12, 17, 22]
        trees = np.random.uniform(0, 1, 5) > 0.2
        trees = np.array(tree_wall)[trees]

        self.chopping = 0

        self.state = self.create_state(agent, monster, trees, self.chopping)

        self.steps = 0
        return self.state.flatten()

    def close(self):
        pass

    def render(self):
        self.render_state(self.state)

    def render_state(self, state):
        agents = np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['AGENT'])[0]
        monsters = np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['MONSTER'])[0]
        trees = list(np.where(state[0:self.world_dim*self.world_dim] == self.OBJECTS['TREE'])[0])

        rendering = '---------------'
        print('STATE = {}'.format(state))

        for i in range(self.world_dim * self.world_dim):
            if i % self.world_dim == 0:
                rendering += '\n'

            if i in agents:
                rendering += ' A '
            elif i in monsters:
                rendering += ' M '
            elif i in trees:
                rendering += ' T '
            else:
                rendering += ' - '

        rendering += '\n'
        rendering += '---------------'
        print(rendering)

    def realistic(self, x):
        x = np.array(x).squeeze()

        agent = list(np.where(x[0:self.world_dim * self.world_dim] == self.OBJECTS['AGENT'])[0])
        monster = list(np.where(x[0:self.world_dim * self.world_dim] == self.OBJECTS['MONSTER'])[0])
        trees = list(np.where(x[0:self.world_dim * self.world_dim] == self.OBJECTS['TREE'])[0])

        if len(agent) != 1:
            return False
        if len(monster) != 1:
            return False
        if len(trees) > 5:
            return False

        return True

    def actionable(self, x, fact):
        monster = np.where(fact == self.OBJECTS['MONSTER'])[0][0]

        return abs(x[monster] == self.OBJECTS['MONSTER'])

    def generate_state_from_json(self, json_dict):
        agent = json_dict['agent']
        monster = json_dict['monster']
        trees = json_dict['trees']

        state = self.create_state(agent, monster, trees, chopping=0)

        return state


