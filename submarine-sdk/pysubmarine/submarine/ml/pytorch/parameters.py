
default_parameters = {
    "output": {
        "save_model_dir": "./output",
        "metric": "roc_auc_score"
    },
    "training": {
        "batch_size": 64,
        "num_epochs": 1,
        "log_steps": 10,
        "num_threads": 0,
        "num_gpus": 0,
        "seed": 42,
        "mode": "distributed",
        "backend": "gloo"
    },
    "model": {
        "name": "ctr.deepfm",
        "kwargs": {
            "out_features": 1,
            "embedding_dim": 256,
            "hidden_units": [400, 400],
            "dropout_rates": [0.2, 0.2]
        }
    },
    "loss": {
        "name": "BCEWithLogitsLoss",
        "kwargs": {}
    },
    "optimizer": {
        "name": "adam",
        "kwargs": {
            "lr": 1e-3
        }
    },
    "resource": {
        "num_cpus": 4,
        "num_gpus": 0,
        "num_threads": 0
    }
}
