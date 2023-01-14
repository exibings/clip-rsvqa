import os
from PIL import Image
import pandas as pd
from transformers import CLIPFeatureExtractor, CLIPTokenizer
import h5py
import numpy as np
from tqdm.auto import tqdm

tokenizer = CLIPTokenizer.from_pretrained("flax-community/clip-rsicd-v2")
feature_extractor = CLIPFeatureExtractor.from_pretrained("flax-community/clip-rsicd-v2")

def jpgOnly(dataset_name: str) -> int:
    """
    Filters all dataset images to .jpg only.

    Args:
        dataset_name (str): Dataset folder name
    """
    count = 0
    if dataset_name == "RSVQAxBEN":
        for subfolder in os.listdir(os.path.join("datasets", dataset_name, "images")):
            for file in os.listdir(os.path.join("datasets", dataset_name, "images", subfolder)):
                if not file.endswith(".jpg"):
                    print("removing", file)
                    os.remove(os.path.join("datasets", dataset_name, "images", file))
                else:
                    count += 1
        return count
    else:
        for file in os.listdir(os.path.join("datasets", dataset_name, "images")):
            if not file.endswith(".jpg"):
                print("removing", file)
                os.remove(os.path.join("datasets", dataset_name, "images", file))
            else:
                count += 1
        return count

def verifyImages(dataset_name: str) -> bool:
    if dataset_name == "RSVQA-LR":
        print("Verifying RSVQA-LR images...")
        return True if len(os.listdir(os.path.join("datasets", "RSVQA-LR", "images"))) == 772 else False
    elif dataset_name == "RSVQA-HR":
        print("Verifying RSVQA-HR images...")
        return True if len(os.listdir(os.path.join("datasets", "RSVQA-HR", "images"))) == 10659 else False
    elif dataset_name == "RSVQAxBEN":
        print("Verifying RSVQAxBEN images...")
        images_checker = {}
        total = 0
        for subfolder in os.listdir(os.path.join("datasets", "RSVQAxBEN", "images")):
            images_checker[subfolder] = len(os.listdir(os.path.join("datasets", "RSVQAxBEN", "images", subfolder)))
            total += images_checker[subfolder]
        return True if total == 590326 else False
    elif dataset_name == "NWPU-Captions":
        print("Verifying NWPU-Captions image...")
        return True if len(os.listdir(os.path.join("datasets", "NWPU-Captions", "images"))) == 31499 else False # this value should be 31500 but the dataset is missing 1 image - airplane_111.jpg
        
def patchImage(img_path: str) -> list:
    """
    Patches the image and returns the patches and original image

    Args:
        img_path (str): Path to the image to be patched

    Returns:
        list: list with the 4 patches generated from the image and the original image - [top_left, top_right, bottom_left, bottom_right, full_image]
    """
    img = Image.open(img_path)
    return [img.crop((0, 0, img.width//2, img.height//2)), img.crop((img.width//2, 0, img.width, img.height//2)),
            img.crop((0, img.height//2, img.width//2, img.height)), img.crop((img.width//2, img.height//2, img.width, img.height)), img]

def encodeDatasetLabels(dataset_name: str, trainDataset: pd.DataFrame = None, validationDataset: pd.DataFrame = None, testDataset: pd.DataFrame = None, testPhiliDataset: pd.DataFrame = None) -> tuple[dict, dict]:
    """
    Create translation dictionaries for the labels from the dataset.

    Translates the labels from text to an id - from 1 to N, where N is the number of possible labels in the dataset.
    """
    if dataset_name == "NWPU-Captions":
        labels = set(list(trainDataset["class"]) + list(testDataset["class"]))
    else: 
        labels = set(list(trainDataset["label"]) + list(validationDataset["label"]) + list(testDataset["label"]))
    if dataset_name == "RSVQA-HR":
        labels.update(testPhiliDataset["label"])
    label2id = {}
    id2label = {}
    count = 0
    for label in labels:
        label2id[label] = count
        id2label[count] = label
        count += 1
    return label2id, id2label

def createDatasetSplit(dataset_name, hfile, split, processed_dataframe, label2id_encodings = None):
    print("Creating", split, "split")
    hfile.create_group(split)
    # tokenize dataset
    if dataset_name in ("RSVQA-LR", "RSVQA-HR", "RSVQAxBEN"):
        max_text_length = min(tokenizer.model_max_length, processed_dataframe.question.str.len().max())
    elif dataset_name == "NWPU-Captions":
        max_text_length = min(tokenizer.model_max_length, processed_dataframe.caption.str.len().max())
    print("\tTokenizing text...")
    if dataset_name in ("RSVQA-LR", "RSVQA-HR", "RSVQAxBEN"):
        processed_dataframe["input_ids"], processed_dataframe["attention_mask"] = processed_dataframe.apply(
            tokenizeText, args=(max_text_length, "question"), result_type="expand", axis="columns").T.values
    elif dataset_name == "NWPU-Captions":
        processed_dataframe["input_ids"], processed_dataframe["attention_mask"] = processed_dataframe.apply(
            tokenizeText, args=(max_text_length, "caption"), result_type="expand", axis="columns").T.values
    # /tokenize dataset
    if dataset_name in ("RSVQA-LR", "RSVQA-HR"):
        hfile[split].create_dataset("img_id", data=processed_dataframe["img_id"])
        hfile[split].create_dataset("category", data=processed_dataframe["category"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("label", data=processed_dataframe["label"])
        hfile[split].create_dataset("question", data=processed_dataframe["question"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("input_ids", (len(processed_dataframe["input_ids"]), max_text_length), np.int32)
        hfile[split].create_dataset("attention_mask", (len(processed_dataframe["attention_mask"]), max_text_length), np.int8)
    elif dataset_name == "RSVQAxBEN":
        print("\tAdding img ids to the dataset...")
        hfile[split].create_dataset("img_id", data=processed_dataframe["img_id"])
        print("\tAdding categories to the dataset...")
        hfile[split].create_dataset("category", data=processed_dataframe["category"], dtype=h5py.special_dtype(vlen=str))       
        print("\tAdding encoded labels to the dataset...")
        hfile[split].create_dataset("label", (len(processed_dataframe["label"]),), np.int64)
        for idx in range(len(processed_dataframe["label"])):
            hfile[split]["label"][idx] = label2id_encodings[processed_dataframe["label"][idx]]
        print("\tAdding questions to the dataset...")
        hfile[split].create_dataset("question", data=processed_dataframe["question"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("input_ids", (len(processed_dataframe["input_ids"]), max_text_length), np.int32)
        hfile[split].create_dataset("attention_mask", (len(processed_dataframe["attention_mask"]), max_text_length), np.int8)
    elif dataset_name == "NWPU-Captions":
        hfile[split].create_dataset("img_id", data=processed_dataframe["img_id"])
        hfile[split].create_dataset("image", data=processed_dataframe["image"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("class", (len(processed_dataframe["class"]),), np.int8)
        for idx in range(len(processed_dataframe["class"])):
            hfile[split]["class"][idx] = label2id_encodings[processed_dataframe["class"][idx]]
        hfile[split].create_dataset("sent_id", data=processed_dataframe["sentid"])
        hfile[split].create_dataset("caption", data=processed_dataframe["caption"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("filtered_caption", data=processed_dataframe["filtered_caption"], dtype=h5py.special_dtype(vlen=str))
        hfile[split].create_dataset("input_ids", (len(processed_dataframe["input_ids"]), max_text_length), np.int32)
        hfile[split].create_dataset("attention_mask", (len(processed_dataframe["attention_mask"]), max_text_length), np.int8)
    progress_bar = tqdm(range(len(processed_dataframe["input_ids"])), desc="Adding processed input ids to the dataset...")
    for idx in range(len(processed_dataframe["input_ids"])):
        hfile[split]["input_ids"][idx] = processed_dataframe["input_ids"][idx]
        progress_bar.update(1)
    progress_bar.close()
    progress_bar = tqdm(range(len(processed_dataframe["attention_mask"])), desc="Adding processed attention masks to the dataset")
    for idx in range(len(processed_dataframe["attention_mask"])):
        hfile[split]["attention_mask"][idx] = processed_dataframe["attention_mask"][idx]
        progress_bar.update(1)
    progress_bar.close()

def tokenizeText(x, max_text_length, column):
    tokenized = tokenizer(x[column], padding="max_length", max_length=max_text_length, return_tensors="np")
    return tokenized["input_ids"], tokenized["attention_mask"]

def count_func(x, cat):
    if cat == 'count':
        try:
            i = int(x)
        except ValueError:
            return x
        if i == 0:
            return '0'
        elif i >= 1 and i <= 10:
            return 'between 1 and 10'
        elif i >= 11 and i <= 100:
            return 'between 11 and 100'
        elif i >= 101 and i <= 1000:
            return 'between 101 and 1000'
        elif i > 1000:
            return 'more than 1000'
    else:
        return x

def area_func(x, cat):
    if cat == 'area':
        area = x.split('m2')[0]
        try:
            area = int(x)
        except ValueError:
            return x

        if area == 0:
            return '0'
        elif area >= 1 and area <= 10:
            return 'between 1 and 10'
        elif area >= 11 and area <= 100:
            return 'between 11 and 100'
        elif area >= 101 and area <= 1000:
            return 'between 101 and 1000'
        elif area > 1000:
            return 'more than 1000'
    else:
        return x
