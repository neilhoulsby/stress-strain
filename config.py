MAX_LEN = 1001
TEST_SIZE = 32
TRAIN_SIZE = 200 - TEST_SIZE
TRAIN_SIZE = 16
DATA_PATH = "data.npz"

base_hparams = {
    "random_seed": 0,
    "model_dir": "/home/neil/proj/stress_strain",
    "physics_decoder": False,
    "max_len": MAX_LEN,
    "num_layers": 6,
    "hidden_dim": 16,
    "mlp_dim": 64,
    "num_heads": 2,
    "dropout_rate": 0.0,
    "attention_dropout_rate": 0.0,
    "deltas_loss_weight": 0.0,
    "physics_loss_weight": 0.0,
    "causal_x": True,
    "batch_size": 16,
    "learning_rate": 1e-2,
    "weight_decay": 0.0,
    "warmup_steps": 100,
    "total_examples": 100_000,
    "eval_freq": 1000,
    "train_size": 200 - 32,
}

# Hyperparameter sweep configuration
sweep_config = {
    "num_layers": [6],
    "hidden_dim": [16],
    "learning_rate": [1e-2],
    "dropout_rate": [0.0],
    "total_examples": [500_000],
    "causal_x": [True],
    "train_size": [200 - 32],
}


def generate_sweep_configs():
    from itertools import product
    
    keys, values = zip(*sweep_config.items())
    sweep_configs = [dict(zip(keys, v)) for v in product(*values)]
    
    for config in sweep_configs:
        hparams = base_hparams.copy()
        hparams.update(config)
        yield hparams