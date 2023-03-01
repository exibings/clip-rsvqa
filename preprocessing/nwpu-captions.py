import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPTokenizer

trainDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "traindf.csv"), sep=",")
validationDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "valdf.csv"), sep=",")
testDataset = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "testdf.csv"), sep=",")
merged_df = pd.concat([trainDataset, validationDataset, testDataset], ignore_index=True)
class2id = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "r"))["class2id"]
trainDataset_extended = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "traindf-extended-captions.csv"), sep=",")
validationDataset_extended = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "traindf-extended-captions.csv"), sep=",")
testDataset_extended = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "traindf-extended-captions.csv"), sep=",")
merged_df_extended = pd.concat([trainDataset, validationDataset, testDataset], ignore_index=True)

trainDataset['caption_length'] = trainDataset.caption.apply(len)
validationDataset['caption_length'] = validationDataset.caption.apply(len)
testDataset['caption_length'] = testDataset.caption.apply(len)
print("train caption length 90th percentile:", trainDataset.caption_length.quantile(0.9))
print("validation caption length 90th percentile:", validationDataset.caption_length.quantile(0.9))
print("test caption length 90th percentile:", testDataset.caption_length.quantile(0.9))

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
trainDataset["caption_length"].hist(ax=axes[0], bins=len(trainDataset.caption_length.unique())).set_title(
    "Train")
validationDataset["caption_length"].hist(ax=axes[1], bins=len(validationDataset.caption_length.unique())).set_title(
    "Validation")
testDataset["caption_length"].hist(ax=axes[2], bins=len(testDataset.caption_length.unique())).set_title(
    "Test")
fig.suptitle("NWPU-Captions Caption Length")
fig.savefig(os.path.join("datasets", "NWPU-Captions", "caption_length_distribution.png"))

trainDataset_extended['caption_length'] = trainDataset_extended.caption.apply(len)
validationDataset_extended['caption_length'] = validationDataset_extended.caption.apply(len)
testDataset_extended['caption_length'] = testDataset_extended.caption.apply(len)
print("train extended caption length 90th percentile:", trainDataset_extended.caption_length.quantile(0.9))
print("validation extended caption length 90th percentile:", validationDataset_extended.caption_length.quantile(0.9))
print("test extended caption length 90th percentile:", testDataset_extended.caption_length.quantile(0.9))

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
trainDataset_extended["caption_length"].hist(ax=axes[0], bins=len(trainDataset_extended.caption_length.unique())).set_title(
    "Train")
validationDataset_extended["caption_length"].hist(ax=axes[1], bins=len(validationDataset_extended.caption_length.unique())).set_title(
    "Validation")
testDataset_extended["caption_length"].hist(ax=axes[2], bins=len(testDataset_extended.caption_length.unique())).set_title(
    "Test")
fig.suptitle("NWPU-Captions Caption Length")
fig.savefig(os.path.join("datasets", "NWPU-Captions", "extended_caption_length_distribution.png"))


image_count = utils.jpgOnly("NWPU-Captions") + 1 # plus 1 to make space for the missing image (airplane_111.jpg)
if utils.verifyImages("NWPU-Captions"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "w") as hfile:
        tokenizer = CLIPTokenizer.from_pretrained("flax-community/clip-rsicd-v2")
        utils.createDatasetSplit("NWPU-Captions", hfile, "train", trainDataset, class2id)
        del trainDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "validation", validationDataset, class2id)
        del validationDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "test", testDataset, class2id)
        del testDataset
        print("Processing class sentences and adding them to the dataset...")
        max_text_length = min(tokenizer.model_max_length, merged_df.class_caption.str.len().max())
        class_sentence_input_ids = hfile.create_dataset("class_sentence_input_ids", (len(class2id), max_text_length), np.int32)
        class_sentence_attention_mask = hfile.create_dataset("class_sentence_attention_mask", (len(class2id), max_text_length), np.int32)
        for _class in class2id:
            tokenized = tokenizer(f"An aerial photograph of a {_class:s}", padding="max_length", max_length=max_text_length, return_tensors="np")
            class_sentence_input_ids[class2id[_class]] = tokenized.input_ids
            class_sentence_attention_mask[class2id[_class]] = tokenized.attention_mask
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "NWPU-Captions", "images"))
        for image_name in images:
            image = utils.feature_extractor(Image.open(os.path.join("datasets", "NWPU-Captions", "images", image_name)), return_tensors="np", resample=Image.Resampling.BILINEAR)
            image_idx = merged_df[merged_df["image"] == image_name]["img_id"].unique()[0]
            pixel_values[image_idx] = image.pixel_values
    
    with h5py.File(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_extended.h5"), "w") as hfile:
        tokenizer = CLIPTokenizer.from_pretrained("saved-models/clip-rscid-v2-extended")

        utils.createDatasetSplit("NWPU-Captions", hfile, "train", trainDataset_extended, class2id, tokenizer)
        del trainDataset_extended
        utils.createDatasetSplit("NWPU-Captions", hfile, "validation", validationDataset_extended, class2id, tokenizer)
        del validationDataset_extended
        utils.createDatasetSplit("NWPU-Captions", hfile, "test", testDataset_extended, class2id, tokenizer)
        del testDataset_extended
        print("Processing class sentences and adding them to the dataset...")
        max_text_length = min(tokenizer.model_max_length, merged_df_extended.class_caption.str.len().max())
        class_sentence_input_ids = hfile.create_dataset("class_sentence_input_ids", (len(class2id), max_text_length), np.int32)
        class_sentence_attention_mask = hfile.create_dataset("class_sentence_attention_mask", (len(class2id), max_text_length), np.int32)
        for _class in class2id:
            tokenized = tokenizer(f"An aerial photograph of a {_class:s}", padding="max_length", max_length=max_text_length, return_tensors="np")
            class_sentence_input_ids[class2id[_class]] = tokenized.input_ids
            class_sentence_attention_mask[class2id[_class]] = tokenized.attention_mask
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "NWPU-Captions", "images"))
        for image_name in images:
            image = utils.feature_extractor(Image.open(os.path.join("datasets", "NWPU-Captions", "images", image_name)), return_tensors="np", resample=Image.Resampling.BILINEAR)
            image_idx = merged_df_extended[merged_df_extended["image"] == image_name]["img_id"].unique()[0]
            pixel_values[image_idx] = image.pixel_values
else:
    print("Missing images.")