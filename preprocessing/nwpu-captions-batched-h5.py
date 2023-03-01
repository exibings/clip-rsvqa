import os
import json
import pandas as pd
from transformers import CLIPTokenizer
import h5py
import utils

dataset_folder =  os.path.join("datasets", "NWPU-Captions")
encodings = json.load(open(os.path.join(dataset_folder, "nwpu_captions_metadata.json"), "r"))
class2id = encodings["class2id"]
id2class = encodings["id2class"]

train_dataset = pd.read_csv(os.path.join(dataset_folder, "batched", "train_batched.csv"))
val_dataset = pd.read_csv(os.path.join(dataset_folder, "batched", "val_batched.csv"))
test_dataset = pd.read_csv(os.path.join(dataset_folder, "batched", "test_batched.csv"))


with h5py.File(os.path.join("datasets", "NWPU-Captions", "batched", "nwpu_captions_batched.h5"), "w") as hfile:
    tokenizer = CLIPTokenizer.from_pretrained("flax-community/clip-rsicd-v2")
    utils.createDatasetSplit("NWPU-Captions", hfile, "train", train_dataset, label2id_encodings=class2id, tokenizer=tokenizer)
    del train_dataset
    utils.createDatasetSplit("NWPU-Captions", hfile, "validation", val_dataset, label2id_encodings=class2id, tokenizer=tokenizer)
    del val_dataset
    utils.createDatasetSplit("NWPU-Captions", hfile, "test", test_dataset, label2id_encodings=class2id, tokenizer=tokenizer)
    del test_dataset