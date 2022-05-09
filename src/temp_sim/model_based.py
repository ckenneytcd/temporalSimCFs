import torch
from omegaconf import omegaconf
import mbrl.util.common as common_util
import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models

from src.envs.chess.chess_gym import ChessGymEnv


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    trial_length = 200
    num_trials = 10
    ensemble_size = 1

    generator = torch.Generator(device=device)

    env = cartpole_env.CartPoleEnv()

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model":
                {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": device,
                    "num_layers": 3,
                    "ensemble_size": ensemble_size,
                    "hid_size": 200,
                    "in_size": "???",
                    "out_size": "???",
                    "deterministic": False,
                    "propagation_method": "fixed_model",
                    # can also configure activation function for GaussianMLP
                    "activation_fn_cfg": {
                        "_target_": "torch.nn.LeakyReLU",
                        "negative_slope": 0.01
                    }
                }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    print(dynamics_model)

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole

    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

    print(model_env)

if __name__ == '__main__':
    main()