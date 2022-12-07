import os
import utils
import h5py
import pandas as pd
import json
import matplotlib.pyplot as plt

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})

trainDataset['question_length'] = trainDataset.question.apply(len)
validationDataset['question_length'] = validationDataset.question.apply(len)
testDataset['question_length'] = testDataset.question.apply(len)
print("train question length 90th percentile:", trainDataset.question_length.quantile(0.9))
print("validation question length 90th percentile:", validationDataset.question_length.quantile(0.9))
print("test question length 90th percentile:", testDataset.question_length.quantile(0.9))

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
trainDataset["question_length"].hist(ax=axes[0], bins=len(trainDataset.question_length.unique())).set_title(
    "Train")
validationDataset["question_length"].hist(ax=axes[1], bins=len(validationDataset.question_length.unique())).set_title(
    "Validation")
testDataset["question_length"].hist(ax=axes[2], bins=len(testDataset.question_length.unique())).set_title(
    "Test")
fig.suptitle("RSVQA-HR Question Length")
fig.savefig(os.path.join("datasets", "RSVQAxBEN", "question_length_distribution.png"))

label2id, id2label = utils.encodeDatasetLabels("RSVQAxBEN", trainDataset, validationDataset, testDataset)
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
with open(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben_metadata.json"), "w") as metadata_file:
    json.dump(metadata, metadata_file)

image_count = utils.jpgOnly("RSVQAxBEN")
if utils.verifyImages("RSVQAxBEN"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQAxBEN", hfile, "train", trainDataset, metadata["label2id"])
        del trainDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "validation", validationDataset, metadata["label2id"])
        del validationDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "test", testDataset, metadata["label2id"])
        del testDataset
else:
    print("Missing images.")