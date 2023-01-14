import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

trainDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "traindf.csv"), sep=",")
validationDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "valdf.csv"), sep=",")
testDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "testdf.csv"), sep=",")
merged_df = pd.concat([trainDataset, validationDataset, testDataset])
class2id = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "r"))["class2id"]

max_text_length = min(utils.tokenizer.model_max_length, merged_df.caption.str.len().max())

image_count = utils.jpgOnly("NWPU-Captions") + 1 # plus 1 to make space for the missing image (airplane_111.jpg)
if utils.verifyImages("NWPU-Captions"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "w") as hfile:
        utils.createDatasetSplit("NWPU-Captions", hfile, "train", trainDataset, class2id)
        del trainDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "validation", validationDataset, class2id)
        del validationDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "test", testDataset, class2id)
        del testDataset
        print("Processing class sentences and adding them to the dataset...")
        class_sentence_input_ids = hfile.create_dataset("class_sentence_input_ids", (len(class2id), max_text_length), np.int32)
        class_sentence_attention_mask = hfile.create_dataset("class_sentence_attention_mask", (len(class2id), max_text_length), np.int32)
        for _class in class2id:
            tokenized = utils.tokenizer(f"An aerial photograph of a {_class:s}", padding="max_length", max_length=max_text_length, return_tensors="np")
            class_sentence_input_ids[class2id[_class]] = tokenized.input_ids
            class_sentence_attention_mask[class2id[_class]] = tokenized.attention_mask
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "NWPU-Captions", "images"))
        for image_name in images:
            image = utils.feature_extractor(Image.open(os.path.join("datasets", "NWPU-Captions", "images", image_name)), return_tensors="np", resample=Image.Resampling.BILINEAR)
            image_idx = merged_df[merged_df["image"] == image_name]["img_id"].unique()[0]
            pixel_values[image_idx] = image.pixel_values
else:
    print("Missing images.")