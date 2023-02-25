import numpy as np
import pandas as pd
from tqdm import tqdm


class Dataset:

    def __init__(self, env, bb_model, dataset_path):
        self.env = env
        self.bb_model = bb_model
        self.dataset_path = dataset_path

        self._dataset = self.generate_dataset(env, bb_model, dataset_path)

    def generate_dataset(self, env, model, dataset_path, n_ep=200):
        try:
            df = pd.read_csv(dataset_path, index_col=False)
            print('Loaded dataset with {} samples'.format(len(df)))
        except FileNotFoundError:
            print('Generating dataset...')
            ds = []

            for i in tqdm(range(n_ep)):
                obs = env.reset()
                done = False
                c = 0
                while (not done) and (c < 50):
                    c += 1
                    if dataset_path == 'datasets/taxi/dataset.csv':
                        desc = env.desc.copy().tolist()
                        out = [[c.decode("utf-8") for c in line] for line in desc]
                        taxi_row, taxi_col, pass_idx, dest_idx = env.decode(obs[0])
                        out.pop(0)
                        out.pop(5)
                        for row in out:
                            for char in row:  
                                if char == '+' or char == '|' or char == ':':
                                    row.remove(char)

                        print(taxi_row, ',', taxi_col)
                        out[taxi_row][taxi_col] = 'T'+out[taxi_row][taxi_col]

                        char_to_int_mapping = {'R':0, 'G':1, 'Y':2, 'B':3, 'T ':4, ' ':5, 'TR':6, 'TG':7, 'TY':8, 'TB':9}
                        int_arr = [obs[0]]
                        for row in out:
                            for char in row:
                                int_arr.append(char_to_int_mapping[char])
                        print(int_arr)
                        ds.append(int_arr)
                        #ds.append([obs[0]])
                    else:
                        ds.append(list(obs))
                    rand = np.random.randint(0, 2)
                    if rand == 0:
                        action = model.predict(obs)
                    else:
                        action = np.random.choice(env.get_actions(obs))
                    if dataset_path == 'datasets/taxi/dataset.csv':
                        obs, rew, done,  _ = env.stepds(action)
                        print(list(env.decode(obs[0])))
                    else:
                        obs, rew, done,  _ = env.step(action)
                        print(obs)
                    
                    # if len(list(obs)) > 1:
                    #     print(i)
                    #     print('\n') 
                    #print(c, rand, action, obs)
            print(ds)     
            # for i in ds:
            #     if len(i) > 1:
            #         print(i)
            #         print('\n')     
            df = pd.DataFrame(ds)
            df = df.drop_duplicates()

            print('Generated {} samples!'.format(len(df)))
            df.to_csv(dataset_path, index=False)

        return df

    def split_dataset(self, frac=0.8):
        train_dataset = self._dataset.sample(frac=0.8, random_state=1)
        test_dataset = self._dataset.drop(train_dataset.index)

        return train_dataset, test_dataset