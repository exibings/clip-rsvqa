import argparse

import torch

from Trainer import Trainer

parser = argparse.ArgumentParser(description="Train and test the CLIPxRSVQA model.")
parser.add_argument("--dataset", metavar="dataset", type=str,
                    help="name of the dataset: RSVQA-LR, RSVQA-HR, RSVQAxBEN", required=True)
parser.add_argument("--epochs", metavar="epochs", type=int,
                    help="maximum number of epochs during training", default=100)
parser.add_argument("--patience", metavar="patience", type=int,
                    help="patience for the training loop. If 0, patience is ignored", default=20)
parser.add_argument("--batch", metavar="batchSize", type=int, help="batch size to be used during training", default=64)

args = parser.parse_args()
args = {"limit_epochs": args.epochs,
        "batch_size": args.batch,
        "patience": args.patience,
        "dataset_name": args.dataset}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args["device"] = device


trainer = Trainer(**args)
print("Trainer is ready.")

print("Starting training session...")
trainer.train()
