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
args = {"limit_epochs": args.epochs,
        "batch_size": args.batch,
        "use_resized_images": args.resized,
        "dataset_name": args.dataset}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args["device"] = device

print(args)

trainer = Trainer(**args)
print("Trainer is ready.\n")

print("Pre-training test sanity check results:\n")
if args["dataset_name"] == "RSVQA-HR":
    results = trainer.test()
    print("\ttest:", results[0], "test phili",  results[1], "\n")
else:
    print("\ttest:", trainer.test())
print("Starting training session...")
trainer.train()
