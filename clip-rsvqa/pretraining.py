import torch
from PreTrainer import PreTrainer
import argparse
import json

parser = argparse.ArgumentParser(description="Pretrain a CLIP Model.")
parser.add_argument("--config", metavar="config", type=str, help="config file with parameters to be used for pretraining.", required=True)
args = parser.parse_args()

config = json.load(open(args.config, "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device
pretrainer = PreTrainer(**config)
print("Pretrainer is ready.")
print("Starting pretraining session...")
pretrainer.run()
