from os.path import exists

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.utils import load_fact


class Task:

    def __init__(self, task_name, env, bb_model, dataset, method, method_name, objs, eval_path):
        self.task_name = task_name
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.method = method
        self.method_name = method_name
        self.eval_path = eval_path
        self.objs = objs

    def run_experiment(self, facts, targets=None):
        print('Running experiment for {} task with {}'.format(self.task_name, self.method_name))
        print('Finding counterfactuals for {} facts'.format(len(facts)))

        # get cfs for facts
        eval_dict = {}
        cnt = 0
        for i in tqdm(range(len(facts))):
            f = facts[i]

            if isinstance(f, dict):
                f = self.env.generate_state_from_json(f)

            if targets is None:
                ts = self.get_targets(f, self.env, self.bb_model)
            else:
                ts = [targets[i]]

            for t in ts:
                cf = self.method.generate_counterfactuals(f, t)

                if cf is None:
                    found = False
                    self.evaluate_cf(f, t, cf, found)
                    continue
                else:
                    found = True
                    e = self.evaluate_cf(f, t, cf, found)

                    try:
                        eval_dict = {obj: (eval_dict[obj] + obj_val) for obj, obj_val in e.items()}
                    except KeyError:
                        eval_dict = e

                    cnt += 1

        print('Average objectives for task = {}, method = {}: {}'.format(
            self.task_name,
            self.method_name,
            list({obj: (obj_val*1.0)/cnt for obj, obj_val in eval_dict.items()}.items())))

    def get_targets(self, f, env, bb_model):
        pred = bb_model.predict(f)
        available_actions = env.get_actions(f)
        targets = [a for a in available_actions if a != pred]

        return targets

    def evaluate_cf(self, f, t, cf, found):
        if not found:
            ind_rew = [0] * self.objs.num_objectives
            objs_names = list(self.objs.lmbdas.keys())
            df = pd.DataFrame([ind_rew], columns=objs_names)

            df['searched_nodes'] = 0
            df['total_reward'] = 0
            df['time'] = 0
            df['cf'] = 0

        else:
            ind_rew = self.objs.get_ind_rews(f, cf.cf_state, t, cf.actions, cf.cummulative_reward)
            total_rew = self.objs.get_reward(f, cf.cf_state, t, cf.actions, cf.cummulative_reward)

            df = pd.DataFrame.from_dict(ind_rew)
            df['searched_nodes'] = cf.searched_nodes
            df['total_reward'] = total_rew
            df['time'] = cf.time

            df['cf'] =  self.env.writable_state(cf.state)


        # add additional parameters like time and tree size
        df['fact'] = list(np.tile(self.env.writable_state(f), (len(df), 1)))
        df['target'] = t
        df['found'] = found

        header = not exists(self.eval_path)
        df.to_csv(self.eval_path, mode='a', header=header)

        return ind_rew