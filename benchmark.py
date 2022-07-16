from entity_gym_rs import multisnake, RustVecEnv
from ragged_buffer import RaggedBufferI64
import numpy as np
import time

env = RustVecEnv(multisnake(num_envs=128, threads=4, board_size=10))
# print(env.reset(None))


env.reset(None)
actions = {
    "Move": RaggedBufferI64.from_flattened(
        np.zeros((2 * len(env), 1), dtype=np.int64),
        np.full(len(env), 2, dtype=np.int64),
    )
}
start = time.time()
N = 20000
for _ in range(N):
    env.act(actions, None)
elapsed = time.time() - start
print(f"Wall time: {elapsed:.2f}s")
print(f"Throughput: {len(env) * N / elapsed / 1000.0:.2f}K steps/s")
