from dataset import *
from train import *


if __name__ == "__main__":
    seed_all(SEED)

    if MODE == "prepare":
        prepare_dataset()

    elif MODE == "train":
        train()