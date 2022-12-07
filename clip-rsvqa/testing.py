import argparse

import torch
import json

from Trainer import Trainer

parser = argparse.ArgumentParser(description="Train and test the CLIPxRSVQA model.")
parser.add_argument("--dataset", metavar="dataset", type=str,
                    help="name of the dataset: RSVQA-LR, RSVQA-HR, RSVQAxBEN", required=True)
parser.add_argument("--batch", metavar="batchSize", type=int, help="batch size to be used during testing", required=True)
parser.add_argument("--model", metavar="model_path", type=str, help=".pth file to be loaded", required=True)

args = parser.parse_args()
args = {"batch_size": args.batch,
        "dataset_name": args.dataset,
        "load_model": True if args.model != None else False,
        "model_path": args.model}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args["device"] = device

trainer = Trainer(**args)
print("Trainer is ready.")

print("Starting testing session...")
print(trainer.test())
