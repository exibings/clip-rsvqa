import os
import json
import pandas as pd
import spacy


nlp = spacy.load('en_core_web_md')

dataset_name = "NWPU-Captions"
file_name = "nwpu_captions.h5"
encodings_file_name = "nwpu_captions_encodings.json"


dataset_folder =  os.path.join("datasets", dataset_name)
encodings = json.load(open(os.path.join(dataset_folder, encodings_file_name), "r"))
if dataset_name == "NWPU-Captions":
    class2id = encodings["class2id"]
    id2class = encodings["id2class"]
else:
    id2label = encodings["id2label"]
    label2id = encodings["label2id"]

dataset = pd.read_csv(os.path.join(dataset_folder, "traindf.csv"), index_col="sentid")
print(dataset.loc[154000]["filtered_caption"])
print(dataset.loc[154001]["filtered_caption"])


