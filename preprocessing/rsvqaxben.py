import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json
from tqdm.auto import tqdm

encodings_path = os.path.join("datasets", "RSVQAxBEN", "rsvqaxben_encodings.json")

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
label2id, id2label = utils.encodeDatasetLabels("RSVQAxBEN", trainDataset, validationDataset, testDataset)
encodings = {"label2id": label2id, "id2label": id2label}
with open(encodings_path, "w") as encodings_file:
    json.dump(encodings, encodings_file)

image_count = utils.jpgOnly("RSVQAxBEN")
if utils.verifyImages("RSVQAxBEN"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQAxBEN", hfile, "train", trainDataset, encodings["label2id"])
        del trainDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "validation", validationDataset, encodings["label2id"])
        del validationDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "test", testDataset, encodings["label2id"])
        del testDataset
else:
    print("Missing images.")