import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json
import matplotlib.pyplot as plt

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})

trainDataset["label"] = trainDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
trainDataset["label"] = trainDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
trainDataset["zero_shot"] = trainDataset.apply(lambda x: f"{x['question']}The answer is {x['label']}", axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
validationDataset["zero_shot"] = validationDataset.apply(lambda x: f"{x['question']}The answer is {x['label']}", axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testDataset["zero_shot"] = testDataset.apply(lambda x: f"{x['question']}The answer is {x['label']}", axis="columns")

trainDataset['question_length'] = trainDataset.question.apply(len)
validationDataset['question_length'] = validationDataset.question.apply(len)
testDataset['question_length'] = testDataset.question.apply(len)
print("train question length 90th percentile:", trainDataset.question_length.quantile(0.9))
print("validation question length 90th percentile:", validationDataset.question_length.quantile(0.9))
print("test question length 90th percentile:", testDataset.question_length.quantile(0.9))

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
trainDataset["question_length"].hist(ax=axes[0], bins=len(trainDataset.question_length.unique())).set_title(
    "Train")
validationDataset["question_length"].hist(ax=axes[1], bins=len(validationDataset.question_length.unique())).set_title(
    "Validation")
testDataset["question_length"].hist(ax=axes[2], bins=len(testDataset.question_length.unique())).set_title(
    "Test")
fig.suptitle("RSVQA-LR Question Length")
fig.savefig(os.path.join("datasets", "RSVQA-LR", "question_length_distribution.png"))

trainDataset['zero_shot_length'] = trainDataset.zero_shot.apply(len)
validationDataset['zero_shot_length'] = validationDataset.zero_shot.apply(len)
testDataset['zero_shot_length'] = testDataset.zero_shot.apply(len)
print("train zero shot prompt max length :", trainDataset.zero_shot_length.max())
print("validation zero shot prompt max length :", validationDataset.zero_shot_length.max())
print("test zero shot prompt max length :", testDataset.zero_shot_length.max())

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
trainDataset["zero_shot_length"].hist(ax=axes[0], bins=len(trainDataset.zero_shot_length.unique())).set_title(
    "Train")
validationDataset["zero_shot_length"].hist(ax=axes[1], bins=len(validationDataset.zero_shot_length.unique())).set_title(
    "Validation")
testDataset["zero_shot_length"].hist(ax=axes[2], bins=len(testDataset.zero_shot_length.unique())).set_title(
    "Test")
fig.suptitle("RSVQA-LR Zero Shot Prompt Length")
fig.savefig(os.path.join("datasets", "RSVQA-LR", "zero_shot_length_distribution.png"))

label2id, id2label = utils.encodeDatasetLabels("RSVQA-LR", trainDataset, validationDataset, testDataset)
trainDataset.replace(label2id, inplace=True)
validationDataset.replace(label2id, inplace=True)
testDataset.replace(label2id, inplace=True)

num_labels = {}
# this is computed with the validation dataset because its a small split that covers all the possible answers
for category, label in zip(validationDataset["category"], validationDataset["label"]):
    try: 
        num_labels[category].add(label)
        num_labels["total"].add(label)  
    except KeyError:
        num_labels[category] = set()
        num_labels["total"] = set() 
        num_labels[category].add(label)
        num_labels["total"].add(label)  
for key in num_labels:
    num_labels[key] = len(num_labels[key])

metadata = {"label2id": label2id, "id2label": id2label, "num_labels": num_labels}
with open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_metadata.json"), "w") as metadata_file:
    json.dump(metadata, metadata_file)

image_count = utils.jpgOnly("RSVQA-LR")
if utils.verifyImages("RSVQA-LR"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQA-LR", hfile, "train", trainDataset)
        del trainDataset
        utils.createDatasetSplit("RSVQA-LR", hfile, "validation", validationDataset)
        del validationDataset
        utils.createDatasetSplit("RSVQA-LR", hfile, "test", testDataset)
        del testDataset
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "RSVQA-LR", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-LR", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
else:
    print("Missing images.")