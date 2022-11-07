from os.path import exists

import pandas as pd

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

    def run_experiment(self):
        print('Running experiment for {} task with {}'.format(self.task_name, self.method_name))

        # get facts
        facts = self.get_facts(self.dataset)

        print('Finding counterfactuals for {} facts'.format(len(facts)))

        # get cfs for facts
        eval_dict = {}
        cnt = 0
        for f in facts:
            print('FACT:')
            self.env.render_state(f)
            targets = self.get_targets(f, self.bb_model)
            for t in targets:
                cf = self.method.generate_counterfactuals(f, t)

                if cf is None:
                    continue

                print('CF:')
                self.env.render_state(cf.cf_state)
                e = self.evaluate_cf(f, t, cf, self.objs, self.eval_path)

                try:
                    eval_dict = {obj: (eval_dict[obj] + obj_val) for obj, obj_val in e.items()}
                except KeyError:
                    eval_dict = e

                cnt += 1

        print('Average objectives for task = {}, method = {}: {}'.format(
            self.task_name,
            self.method_name,
            list({obj: (obj_val*1.0)/cnt for obj, obj_val in eval_dict.items()}.items())))

    def get_facts(self, dataset):
        fact_file = 'fact/{}.json'.format(self.task_name)

        json_facts, targets = load_fact(fact_file)

        facts = []
        for i in range(len(json_facts)):
            json_fact, target = json_facts[i], targets[i]
            fact = self.env.generate_state_from_json(json_fact)
            facts.append(fact)

        # facts = dataset.values

        return facts

    def get_targets(self, f, bb_model):
        return [5]

    def evaluate_cf(self, f, t, cf, obj, eval_path):
        ind_rew = obj.get_ind_rews(f, cf.cf_state, t, cf.actions, cf.cummulative_reward)
        df = pd.DataFrame.from_dict(ind_rew)

        # add additional parameters like time and tree size
        df['searched_nodes'] = cf.searched_nodes
        df['time'] = cf.time

        header = not exists(eval_path)
        df.to_csv(eval_path, mode='a', header=header)

        return ind_rew