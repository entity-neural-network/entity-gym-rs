from entity_gym_rs import multisnake, RustVecEnv
import bevy_snake_enn
from ragged_buffer import RaggedBufferI64
import numpy as np
import time

NUM_ENVS = 2048
THREADS = 4

#NUM_ENVS = 128
#THREADS = 4

def benchmark(env: RustVecEnv, num_steps: int) -> float:
    env.reset(None)
    actions = {
        "Move": RaggedBufferI64.from_flattened(
            np.zeros((1 * len(env), 1), dtype=np.int64),
            np.full(len(env), 1, dtype=np.int64),
        )
    }
    start = time.time()
    for _ in range(num_steps):
        env.act(actions, None)
    elapsed = time.time() - start
    print(f"{NUM_ENVS=} {THREADS=}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Throughput: {len(env) * num_steps / elapsed / 1000.0:.2f}K steps/s")


print("Benchmarking multisnake...")
low_level_env = RustVecEnv(
    multisnake(num_envs=NUM_ENVS, threads=THREADS, board_size=10)
)
benchmark(low_level_env, num_steps=2000)

print("Benchmarking bevy_snake_enn...")
bevy_env = RustVecEnv(
    bevy_snake_enn.create_env(
        bevy_snake_enn.Config(),
        num_envs=NUM_ENVS,
        threads=THREADS,
        first_env_index=0,
    )
)
benchmark(bevy_env, num_steps=50)
