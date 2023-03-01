import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testPhiliDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "testdf_phili.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
trainDataset["label"] = trainDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
trainDataset["label"] = trainDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testPhiliDataset["label"] = testPhiliDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testPhiliDataset["label"] = testPhiliDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
label2id, id2label = utils.encodeDatasetLabels("RSVQA-HR", trainDataset, validationDataset, testDataset, testPhiliDataset)

trainZeroDict = {"img_id": [], "prompt": [], "correct_prompt": [], "category": []}
for _, row in trainDataset.iterrows():
    trainZeroDict["img_id"] += [row["img_id"]]*len(label2id)
    trainZeroDict["category"] += [row["category"]]*len(label2id)
    for label in label2id:
        trainZeroDict["correct_prompt"].append(1 if label == row["label"] else 0)
        prompt = f"{row['question']} The answer is {label}"
        trainZeroDict["prompt"].append(prompt)
validationZeroDict = {"img_id": [], "prompt": [], "correct_prompt": [], "category": []}
for _, row in validationDataset.iterrows():
    validationZeroDict["img_id"] += [row["img_id"]]*len(label2id)
    validationZeroDict["category"] += [row["category"]]*len(label2id)
    for label in label2id:
        validationZeroDict["correct_prompt"].append(1 if label == row["label"] else 0)
        prompt = f"{row['question']} The answer is {label}"
        validationZeroDict["prompt"].append(prompt)
testZeroDict = {"img_id": [], "prompt": [], "correct_prompt": [], "category": []}
for _, row in testDataset.iterrows():
    testZeroDict["img_id"] += [row["img_id"]]*len(label2id)
    testZeroDict["category"] += [row["category"]]*len(label2id)
    for label in label2id:
        testZeroDict["correct_prompt"].append(1 if label == row["label"] else 0)
        prompt = f"{row['question']} The answer is {label}"
        testZeroDict["prompt"].append(prompt)
testPhiliZeroDict = {"img_id": [], "prompt": [], "correct_prompt": [], "category": []}
for _, row in testDataset.iterrows():
    testPhiliZeroDict["img_id"] += [row["img_id"]]*len(label2id)
    testPhiliZeroDict["category"] += [row["category"]]*len(label2id)
    for label in label2id:
        testPhiliZeroDict["correct_prompt"].append(1 if label == row["label"] else 0)
        prompt = f"{row['question']} The answer is {label}"
        testPhiliZeroDict["prompt"].append(prompt)

trainZeroDataset = pd.DataFrame(trainZeroDict)
del trainZeroDict
validationZeroDataset = pd.DataFrame(validationZeroDict)
del validationZeroDict
testZeroDataset = pd.DataFrame(testZeroDict)
del testZeroDict
testPhiliZeroDataset = pd.DataFrame(testPhiliZeroDict)
del testPhiliZeroDict
trainZeroDataset.to_csv("datasets/RSVQA-HR/traindf-zero.csv", index=False)
validationZeroDataset.to_csv("datasets/RSVQA-HR/valdf-zero.csv", index=False)
testZeroDataset.to_csv("datasets/RSVQA-HR/testdf-zero.csv", index=False)
testPhiliZeroDataset.to_csv("datasets/RSVQA-HR/testdf_phili-zero.csv", index=False)
trainDataset['question_length'] = trainDataset.question.apply(len)
validationDataset['question_length'] = validationDataset.question.apply(len)
testDataset['question_length'] = testDataset.question.apply(len)
testPhiliDataset['question_length'] = testPhiliDataset.question.apply(len)
print("train question length 90th percentile:", trainDataset.question_length.quantile(0.9))
print("validation question length 90th percentile:", validationDataset.question_length.quantile(0.9))
print("test question length 90th percentile:", testDataset.question_length.quantile(0.9))
print("test phili question length 90th percentile:", testPhiliDataset.question_length.quantile(0.9))

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
trainDataset["question_length"].hist(ax=axes[0], bins=len(trainDataset.question_length.unique())).set_title(
    "Train")
validationDataset["question_length"].hist(ax=axes[1], bins=len(validationDataset.question_length.unique())).set_title(
    "Validation")
testDataset["question_length"].hist(ax=axes[2], bins=len(testDataset.question_length.unique())).set_title(
    "Test")
testPhiliDataset["question_length"].hist(ax=axes[3], bins=len(testPhiliDataset.question_length.unique())).set_title(
    "Test Philadelphia")
fig.suptitle("RSVQA-HR Question Length")
fig.savefig(os.path.join("datasets", "RSVQA-HR", "question_length_distribution.png"))

trainZeroDataset['prompt_length'] = trainZeroDataset.prompt.apply(len)
validationZeroDataset['prompt_length'] = validationZeroDataset.prompt.apply(len)
testZeroDataset['prompt_length'] = testZeroDataset.prompt.apply(len)
testPhiliZeroDataset['prompt_length'] = testPhiliZeroDataset.prompt.apply(len)
print("train zero shot prompt max length :", trainZeroDataset.prompt_length.max())
print("validation zero shot prompt max length :", validationZeroDataset.prompt_length.max())
print("test zero shot prompt max length :", testZeroDataset.prompt_length.max())
print("test phili zero shot prompt max length :", testPhiliZeroDataset.prompt_length.max())

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
trainZeroDataset["prompt_length"].hist(ax=axes[0], bins=len(trainZeroDataset.prompt_length.unique())).set_title(
    "Train")
validationZeroDataset["prompt_length"].hist(ax=axes[1], bins=len(validationZeroDataset.prompt_length.unique())).set_title(
    "Validation")
testZeroDataset["prompt_length"].hist(ax=axes[2], bins=len(testZeroDataset.prompt_length.unique())).set_title(
    "Test")
testPhiliZeroDataset["prompt_length"].hist(ax=axes[2], bins=len(testPhiliZeroDataset.prompt_length.unique())).set_title(
    "Test")

trainDataset.replace(label2id, inplace=True)
validationDataset.replace(label2id, inplace=True)
testDataset.replace(label2id, inplace=True)
testPhiliDataset.replace(label2id, inplace=True)

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
with open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_metadata.json"), "w") as metadata_file:
    json.dump(metadata, metadata_file)

image_count = utils.jpgOnly("RSVQA-HR")
if utils.verifyImages("RSVQA-HR"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQA-HR", hfile, "train", trainDataset)
        utils.createDatasetSplit("RSVQA-HR", hfile, "validation", validationDataset)
        utils.createDatasetSplit("RSVQA-HR", hfile, "test", testDataset)
        utils.createDatasetSplit("RSVQA-HR", hfile, "test_phili", testPhiliDataset)
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "RSVQA-HR", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-HR", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_extended.h5"), "w") as hfile:
        tokenizer = CLIPTokenizer.from_pretrained("saved-models/clip-rscid-v2-extended")
        utils.createDatasetSplit("RSVQA-HR", hfile, "train", trainDataset, tokenizer=tokenizer)
        del trainDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "validation", validationDataset, tokenizer=tokenizer)
        del validationDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "test", testDataset,tokenizer=tokenizer)
        del testDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "test_phili", testPhiliDataset, tokenizer=tokenizer)
        del testPhiliDataset
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "RSVQA-HR", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-HR", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
else:
    print("Missing images.")