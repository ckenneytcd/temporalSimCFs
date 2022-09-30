from src.baselines.genetic_baseline import GeneticBaseline
from src.cfs.bb_model import BBModel
from src.cfs.dataset import Dataset
from src.cfs.util import load_fact
from src.envs.gridworld import Gridworld


def main():
    task_name = 'gridworld'
    env = Gridworld()
    model_path = 'trained_models/{}'.format(task_name)
    fact_file = 'fact/{}.json'.format(task_name)

    bb_model = BBModel(env, model_path)
    dataset = Dataset(env, bb_model)

    baseline_genetic = GeneticBaseline(env, bb_model, dataset._dataset)
    baseline_genetic_vae = GeneticBaseline(env, bb_model, dataset._dataset, proximity='vae')

    methods = [baseline_genetic_vae]
    method_names = ['Genetic baseline + VAE']

    json_fact, target = load_fact(fact_file)
    fact = env.generate_state_from_json(json_fact)

    for i, m in enumerate(methods):
        print(method_names[i])
        cfs = m.generate_counterfactuals(fact, target)

        print('Found {} counterfactuals'.format(cfs.shape[0]))
        print('Printing the best 10')
        for i, cf in enumerate(cfs[:10, :]):
            env.render_state(cf)

if __name__ == '__main__':
    main()
