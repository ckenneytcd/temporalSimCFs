class Task:

    def __init__(self, task_name, bb_model, dataset, method, method_name, eval_path):
        self.task_name = task_name
        self.bb_model = bb_model
        self.dataset = dataset
        self.method = method
        self.method_name = method_name
        self.eval_path = eval_path

    def run_experiment(self):
        print('Running experiment for {} task'.format(self.task_name))

        # get facts
        facts = self.get_facts(self.dataset)

        print('Finding counterfactuals for {} facts'.format(len(facts)))

        # get cfs for facts
        eval_dict = {}
        cnt = 0
        for f in facts:
            targets = self.get_targets(f, self.bb_model)
            for t in targets:
                cfs = self.method.generate_counterfactuals(f, t)
                e = self.evaluate_cfs(cfs, self.eval_path)
                try:
                    eval_dict = {eval_dict[obj] + obj_val for obj, obj_val in e.items()}
                except KeyError:
                    eval_dict = e
                cnt += 1

        print('Average objectives for task = {}, method = {}: {}'.format(
            self.task_name,
            self.method_name,
            *[(obj, (obj_val*1.0)/cnt) for obj, obj_val in eval_dict.items()]))

    def get_facts(self, dataset):
        pass

    def get_targets(self, f, bb_model):
        pass

    def evaluate_cfs(self, cfs, eval_path):
        pass