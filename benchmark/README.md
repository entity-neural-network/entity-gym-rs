# Training throughput benchmark

This achieves ~120K samples per second on a 2080Ti and Ryzen Threadripper 2990WX:

```
poetry install
poetry run pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
poetry run maturin develop --release --features=python --manifest-path=../Cargo.toml

poetry run python train.py optim.bs=16384 rollout.num_envs=2048 net.n_layer=1
```
