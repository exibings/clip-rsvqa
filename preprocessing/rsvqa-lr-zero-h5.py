import os
import json
import pandas as pd
from transformers import CLIPTokenizer
import h5py
import utils
import numpy as np
from PIL import Image

dataset_folder =  os.path.join("datasets", "RSVQA-LR")

train_dataset = pd.read_csv(os.path.join(dataset_folder, "traindf-zero.csv"))
val_dataset = pd.read_csv(os.path.join(dataset_folder, "valdf-zero.csv"))
test_dataset = pd.read_csv(os.path.join(dataset_folder, "testdf-zero.csv"))

image_count = utils.jpgOnly("RSVQA-LR")
with h5py.File(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_zero.h5"), "w") as hfile:
    tokenizer = CLIPTokenizer.from_pretrained("saved-models/clip-rs")

    split = "train"
    print(f"Creating {split} split")
    hfile.create_group(split)
    max_text_length = min(tokenizer.model_max_length, train_dataset.prompt.str.len().max())
    train_dataset["input_ids"], train_dataset["attention_mask"] = train_dataset.apply(
            utils.tokenizeText, args=(max_text_length, "prompt", tokenizer), result_type="expand", axis="columns").T.values
    hfile[split].create_dataset("img_id", data=train_dataset["img_id"])
    hfile[split].create_dataset("category", data=train_dataset["category"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("correct_prompt", data=train_dataset["correct_prompt"])
    hfile[split].create_dataset("prompt", data=train_dataset["prompt"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("input_ids", (len(train_dataset["input_ids"]), max_text_length), np.int32)
    hfile[split].create_dataset("attention_mask", (len(train_dataset["attention_mask"]), max_text_length), np.int8)

    for idx in range(len(train_dataset["input_ids"])):
        hfile[split]["input_ids"][idx] = train_dataset["input_ids"][idx]
    for idx in range(len(train_dataset["attention_mask"])):
        hfile[split]["attention_mask"][idx] = train_dataset["attention_mask"][idx]

    split = "validation"
    print(f"Creating {split} split")
    hfile.create_group(split)
    max_text_length = min(tokenizer.model_max_length, val_dataset.prompt.str.len().max())
    val_dataset["input_ids"], val_dataset["attention_mask"] = val_dataset.apply(
            utils.tokenizeText, args=(max_text_length, "prompt", tokenizer), result_type="expand", axis="columns").T.values
    hfile[split].create_dataset("img_id", data=val_dataset["img_id"])
    hfile[split].create_dataset("category", data=val_dataset["category"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("correct_prompt", data=val_dataset["correct_prompt"])
    hfile[split].create_dataset("prompt", data=val_dataset["prompt"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("input_ids", (len(val_dataset["input_ids"]), max_text_length), np.int32)
    hfile[split].create_dataset("attention_mask", (len(val_dataset["attention_mask"]), max_text_length), np.int8)
    for idx in range(len(val_dataset["input_ids"])):
        hfile[split]["input_ids"][idx] = val_dataset["input_ids"][idx]
    for idx in range(len(val_dataset["attention_mask"])):
        hfile[split]["attention_mask"][idx] = val_dataset["attention_mask"][idx]
    
    split = "test"
    print(f"Creating {split} split")
    hfile.create_group(split)
    max_text_length = min(tokenizer.model_max_length, test_dataset.prompt.str.len().max())
    test_dataset["input_ids"], test_dataset["attention_mask"] = test_dataset.apply(
            utils.tokenizeText, args=(max_text_length, "prompt", tokenizer), result_type="expand", axis="columns").T.values
    hfile[split].create_dataset("img_id", data=test_dataset["img_id"])
    hfile[split].create_dataset("correct_prompt", data=test_dataset["correct_prompt"])
    hfile[split].create_dataset("category", data=test_dataset["category"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("prompt", data=test_dataset["prompt"], dtype=h5py.special_dtype(vlen=str))
    hfile[split].create_dataset("input_ids", (len(test_dataset["input_ids"]), max_text_length), np.int32)
    hfile[split].create_dataset("attention_mask", (len(test_dataset["attention_mask"]), max_text_length), np.int8)
    for idx in range(len(test_dataset["input_ids"])):
        hfile[split]["input_ids"][idx] = test_dataset["input_ids"][idx]
    for idx in range(len(test_dataset["attention_mask"])):
        hfile[split]["attention_mask"][idx] = test_dataset["attention_mask"][idx]

    pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
    images = os.listdir(os.path.join("datasets", "RSVQA-LR", "images"))
    for image_idx in range(len(images)):
        image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-LR", "images", images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
        pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values