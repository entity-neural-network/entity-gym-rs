import hyperstate
from enn_trainer import TrainConfig, State, init_train_state, train, EnvConfig
from entity_gym.env import VecEnv
from entity_gym_rs import multisnake, RustVecEnv


def create_snake_vec_env(
    cfg: EnvConfig, num_envs: int, num_processes: int, first_env_index: int
) -> VecEnv:
    return RustVecEnv(
        multisnake(num_envs=num_envs, threads=num_processes, board_size=10)
    )


@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=create_snake_vec_env)


if __name__ == "__main__":
    main()
