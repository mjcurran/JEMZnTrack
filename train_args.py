import dataclasses

@dataclasses.dataclass
class train_args:
    norm: str = None
    load_path: str = "./experiment"
    experiment: str = "energy-models"
    dataset: str = "./dataset"
    n_classes: int = 10
    n_steps: int = 20
    width: int = 10
    depth: int = 28
    sigma: float = 0.3
    data_root: str = "./dataset" 
    seed: int = 123456
    lr: float = 1e-4
    clf_only: bool = False
    labels_per_class: int = -1
    batch_size: int = 64
    n_epochs: int = 10
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    save_dir: str = "./experiment"
    ckpt_every: int = 1
    eval_every: int = 11
    print_every: int = 100
    print_to_log: bool = False
    n_valid: int = 5000