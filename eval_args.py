import dataclasses

@dataclasses.dataclass
class eval_args():
    
    experiment: str = "energy_model"
    dataset: str = "cifar_test"
    n_steps: int = 20
    width: int = 10
    depth: int = 28
    sigma: float = .03
    data_root: str = "./dataset"
    seed: int = 123456
    norm: str = None
    save_dir: str = "./experiment"
    print_to_log: bool = False
    uncond: bool = False
    load_path: str = "./experiment"