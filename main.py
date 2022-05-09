from src.envs.chess.chess_gym import ChessGymEnv
from src.envs.chess.chess_task import ChessTask
from src.envs.chess.chess_util import from_board_to_fen, ACTION_NAMES, COL_NAMES, SQUARE_NAMES
from src.baselines.DICE import DICE
from src.baselines.face import FACE
import pandas as pd
from src.baselines.growing_spheres import GrowingSpheres

from src.baselines.relace import ReLACE
from src.util import seed_everything
from src.temp_sim.tsa import TSA


def main():
    seed_everything()
    # define paths
    dataset_name = 'chess'
    model_path = 'engines/{}/stockfish_14.1/stockfish_14.1.exe'.format(dataset_name)
    dataset_path = 'data/{}/dataset.csv'.format(dataset_name)
    json_file_path = 'data/{}/puzzles.json'.format(dataset_name)

    # define task
    if dataset_name == 'chess':
        task = ChessTask(model_path, dataset_path)

    print('------ DATASET SAMPLE ------')
    print(task.dataset.head())

    # get factuals from the data to generate counterfactual examples
    factuals = task.load_factuals(json_file_path)
    fact = factuals.head(1)  # take the first factual for simplicity
    fact = fact[SQUARE_NAMES + ['player']]  # remove the label

    print('Factual:')
    task.print_instance(fact.values.squeeze())
    print('Factual fen: {}'.format(from_board_to_fen(fact.values[0])))

    # define counterfactual action
    target_action = 'c4c8'

    # define which are mutable features
    mutable_features = COL_NAMES
    mutable_features.remove('player')
    mutable_features.remove('best_move')

    chess_env = ChessGymEnv()

    # define feature range (for DICE)
    range_dict = {f: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for f in mutable_features}

    # define counterfactual methods
    dice = DICE(task.bb_model, task.dataset, target_action=ACTION_NAMES.index(target_action), cont_features=[],
                outcome='best_move', mutable_features=mutable_features, range_dict=range_dict)
    growing_spheres = GrowingSpheres(task.bb_model, target_action)
    face = FACE(task.dataset[SQUARE_NAMES + ['player']], bb_model=task.bb_model, target_action=target_action, immutable_keys=['player'])
    TSA = TSA(chess_env, task.bb_model, target_action=target_action)

    methods = [face, dice, growing_spheres, TSA]
    method_names = ['FACE', 'DICE', 'Growing Spheres', 'TSA']

    # generate counterfactual examples
    for i, m in enumerate(methods):
        print('-------------------------------------')
        print('Model = {}'.format(method_names[i]))
        print('-------------------------------------')

        # generate counterfactuals
        cfs = m.get_counterfactuals(fact)

        print('Factual:')
        task.print_instance(fact.values.squeeze())

        if method_names[i] == 'DICE':
            cfs = cfs[0].cf_examples_list[0].final_cfs_df
            if cfs is None:
                continue

        if isinstance(cfs, pd.DataFrame):
            cfs = list(cfs.values)

        # print counterfactuals
        n_cfs = len(cfs)
        print('Found {} counterfactuals.'.format(n_cfs))
        if n_cfs == 0:
            continue

        print('Counterfactuals:')
        for i in range(n_cfs):
            cf_board = cfs[i]
            cf_fen = from_board_to_fen(cf_board.squeeze())
            print(cf_fen)

    print('Finished!')
    task.bb_model.engine.quit()


if __name__ == '__main__':
    main()

