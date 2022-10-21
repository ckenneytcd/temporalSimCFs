from src.baselines.genetic_baseline import GeneticBaseline
from src.models.bb_model import BBModel
from src.models.dataset import Dataset
from src.envs.gridworld import Gridworld
from src.temporal_cf.monte_carlo_cfsearch import MCTSSearch
from src.utils.utils import seed_everything, load_fact


def main():
    seed_everything(seed=1)

    task_name = 'gridworld'
    env = Gridworld()
    model_path = 'trained_models/{}'.format(task_name)
    fact_file = 'fact/{}.json'.format(task_name)

    bb_model = BBModel(env, model_path)
    dataset = Dataset(env, bb_model)

    baseline_genetic_vae = GeneticBaseline(env, bb_model, dataset._dataset)
    mcts_search = MCTSSearch(env, bb_model, dataset._dataset)

    methods = [mcts_search]
    method_names = ['MCTS', 'Genetic baseline + VAE']

    json_facts, targets = load_fact(fact_file)

    for i in range(len(json_facts)):
        json_fact, target = json_facts[i], targets[i]
        fact = env.generate_state_from_json(json_fact)

        print('FACT')
        env.render_state(fact)

        for i, m in enumerate(methods):
            print('-------------------------------------------------------------')
            print('Generating counterfactuals with: {}'.format(method_names[i]))
            cfs = m.generate_counterfactuals(fact, target)

            if len(cfs):
                n_cfs = len(cfs['cf'])
                print('Found {} counterfactuals'.format(n_cfs))

                for i in range(n_cfs):
                    env.render_state(cfs['cf'][i])
                    for k, v in cfs.items():
                        print('{} = {}'.format(k, v[i]))
            else:
                print('Found no counterfactuals')

            print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
