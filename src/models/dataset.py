import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, env, bb_model):
        self.env = env
        self.bb_model = bb_model

        self._dataset = self.generate_dataset(env, bb_model)

    def generate_dataset(self, env, model, n_ep=1000):
        print('Generating dataset...')
        ds = []

        for i in range(n_ep):
            if i % 1000 == 0:
                print('Generated {} samples'.format(i))

            obs = env.reset()

            done = False
            while not done:
                ds.append(list(obs))
                rand = np.random.randint(0, 2)
                if rand == 0:
                    action = model.predict(obs)
                else:
                    action = env.action_space.sample()

                obs, rew, done,  _ = env.step(action)

        df = pd.DataFrame(ds)
        df = df.drop_duplicates()

        print('Generated {} samples!'.format(len(df)))
        return df