# Configuration and hyperparameters

MAX_LEN = 1001
TEST_SIZE = 32
TRAIN_SIZE = 200 - TEST_SIZE

hparams = {
    "random_seed": 0,
    "model_dir": "/tmp/test",
    "physics_decoder": False,
    "max_len": MAX_LEN,
    "num_layers": 4,
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
    "total_steps": 1000,
    "eval_freq": 500,
}

# Data configuration
DATA_PATH = "data.npz"
