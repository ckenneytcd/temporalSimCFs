import chess
import torch

from src.envs.chessenv import ChessEnv
from src.optimization.autoenc import AutoEncoder
from src.optimization.genetic_baseline import GeneticBaseline
from src.models.bb_model import BBModel
from src.models.dataset import Dataset
from src.envs.gridworld import Gridworld
from src.objectives.baseline_objs import BaselineObjectives
from src.objectives.rl_objs import RLObjs
from src.tasks.task import Task
from src.optimization.monte_carlo_cfsearch import MCTSSearch
from src.utils.utils import seed_everything


def main():
    seed_everything(seed=1)

    task_name = 'gridworld'
    if task_name == 'gridworld':
        env = Gridworld()
    elif task_name == 'chess':
        engine_path = 'trained_models/stockfish_15.exe'
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        env = ChessEnv(engine)

    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    fact_file = 'fact/{}.json'.format(task_name)

    # define models
    bb_model = BBModel(env, model_path)
    dataset = Dataset(env, bb_model)
    train_dataset, test_dataset = dataset.split_dataset(frac=0.8)
    vae = AutoEncoder(layers=[env.state_dim, 128, 8])
    vae.fit(train_dataset, test_dataset)
    enc_data = vae.encode(torch.tensor(dataset._dataset.values))[0]

    # define objectives
    baseline_obj = BaselineObjectives(env, bb_model, vae, enc_data, env.state_dim)
    rl_obj = RLObjs(env, bb_model, max_actions=10)

    # get facts
    n_facts = 200
    facts = dataset._dataset.sample(n=n_facts).values

    # define methods
    BO_GEN = GeneticBaseline(env, bb_model, dataset._dataset, baseline_obj)
    BO_MCTS = MCTSSearch(env, bb_model, dataset._dataset, baseline_obj)
    RL_MCTS = MCTSSearch(env, bb_model, dataset._dataset, rl_obj)

    # method names
    methods = [BO_MCTS, RL_MCTS]
    method_names = ['BO_MCTS', 'RL_MCTS']

    for i, m in enumerate(methods):
        eval_path = 'eval/{}/{}/rl_obj_results'.format(task_name, method_names[i])
        task = Task(task_name, facts, env, bb_model, dataset, m, method_names[i], rl_obj, eval_path)
        task.run_experiment()

if __name__ == '__main__':
    main()
