import torch
from PreTrainer import PreTrainer

pretrainer = PreTrainer(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Trainer is ready.")
pretrainer.run()
