import argparse
import torch
import json
from Trainer import Trainer

parser = argparse.ArgumentParser(description="Train and test the CLIPxRSVQA model.")
parser.add_argument("--config", metavar="config", type=str, help="config file with parameters to be used for training.", required=True)
args = parser.parse_args()

config = json.load(open(args.config, "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device
trainer = Trainer(**config)
print("Trainer is ready.")
print("Starting training session...")
trainer.run()
