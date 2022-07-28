import argparse

import torch

from Trainer import Trainer

parser = argparse.ArgumentParser(description="Train and test the CLIPxRSVQA model.")
parser.add_argument("--dataset", metavar="dataset", type=str,
                    help="name of the dataset: RSVQA-LR, RSVQA-HR, RSVQAxBEN")
parser.add_argument("--epochs", metavar="epochs", type=int, help="maximum number of epochs during training", default=25)
parser.add_argument("--batch", metavar="batchSize", type=int, help="batch size to be used during training", default=64)
parser.add_argument("--resized", metavar="resized", type=bool, help="use resized dataset images", default=False)

args = parser.parse_args()
args = {"limitEpochs": args.epochs,
        "batchSize": args.batch,
        "useResizedImages": args.resized,
        "datasetName": args.dataset}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(args, "device", device)


trainer = Trainer(**args)
print(args)
# trainer.train()
