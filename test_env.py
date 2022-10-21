from src.envs.gridworld import Gridworld
from src.utils.utils import seed_everything


def main():
    seed_everything(seed=1)
    env = Gridworld()

    done = False
    env.reset()
    env.render()
    while not done:
        action = env.action_space.sample()
        print('Action = {}'.format(action))
        obs, rew, done, _ = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
