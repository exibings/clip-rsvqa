import os
import json
import pandas as pd
import utils
import spacy
from itertools import combinations

nlp = spacy.load('en_core_web_md')
included_tags = {"NOUN", "ADJ", "NUM"}

trainDataset = {"class": [], "image": [], "img_id": [], "caption": [], "filtered_caption": [], "sentid": []}
testDataset = {"class": [], "image": [], "img_id": [], "caption": [],  "filtered_caption": [], "sentid": []}
valDataset = {"class": [], "image": [], "img_id": [], "caption": [],  "filtered_caption": [], "sentid": []}
with open(os.path.join("datasets", "NWPU-Captions", "dataset_nwpu.json"), "r") as nwpu_file:
    nwpu_data = json.load(nwpu_file)
    
    for label in nwpu_data:
        for row in nwpu_data[label]:
            if row["split"] == "train":
                if row["filename"] == "airplane_111.jpg":
                    continue
                trainDataset["class"].extend(label for _ in range(5))
                trainDataset["image"].extend(row["filename"] for _ in range(5))
                trainDataset["img_id"].extend(row["imgid"] for _ in range(5))
                trainDataset["caption"].append(row["raw"])
                trainDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw"]) if t.pos_ in included_tags]))
                trainDataset["caption"].append(row["raw_1"])
                trainDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_1"]) if t.pos_ in included_tags]))
                trainDataset["caption"].append(row["raw_2"])
                trainDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_2"]) if t.pos_ in included_tags]))
                trainDataset["caption"].append(row["raw_3"])
                trainDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_3"]) if t.pos_ in included_tags]))
                trainDataset["caption"].append(row["raw_4"])
                trainDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_4"]) if t.pos_ in included_tags]))
                trainDataset["sentid"].append(row["sentids"])
                trainDataset["sentid"].append(row["sentids_1"])
                trainDataset["sentid"].append(row["sentids_2"])
                trainDataset["sentid"].append(row["sentids_3"])
                trainDataset["sentid"].append(row["sentids_4"])
            
            elif row["split"] == "val":
                if row["filename"] == "airplane_111.jpg":
                    continue
                valDataset["class"].extend(label for _ in range(5))
                valDataset["image"].extend(row["filename"] for _ in range(5))
                valDataset["img_id"].extend(row["imgid"] for _ in range(5))
                valDataset["caption"].append(row["raw"])
                valDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw"]) if t.pos_ in included_tags]))
                valDataset["caption"].append(row["raw_1"])
                valDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_1"]) if t.pos_ in included_tags]))
                valDataset["caption"].append(row["raw_2"])
                valDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_2"]) if t.pos_ in included_tags]))
                valDataset["caption"].append(row["raw_3"])
                valDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_3"]) if t.pos_ in included_tags]))
                valDataset["caption"].append(row["raw_4"])
                valDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_4"]) if t.pos_ in included_tags]))
                valDataset["sentid"].append(row["sentids"])
                valDataset["sentid"].append(row["sentids_1"])
                valDataset["sentid"].append(row["sentids_2"])
                valDataset["sentid"].append(row["sentids_3"])
                valDataset["sentid"].append(row["sentids_4"])

            elif row["split"] == "test":
                if row["filename"] == "airplane_111.jpg":
                    continue
                testDataset["class"].extend(label for _ in range(5))
                testDataset["image"].extend(row["filename"] for _ in range(5))
                testDataset["img_id"].extend(row["imgid"] for _ in range(5))
                testDataset["caption"].append(row["raw"])
                testDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw"]) if t.pos_ in included_tags]))
                testDataset["caption"].append(row["raw_1"])
                testDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_1"]) if t.pos_ in included_tags]))
                testDataset["caption"].append(row["raw_2"])
                testDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_2"]) if t.pos_ in included_tags]))
                testDataset["caption"].append(row["raw_3"])
                testDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_3"]) if t.pos_ in included_tags]))
                testDataset["caption"].append(row["raw_4"])
                testDataset["filtered_caption"].append(' '.join([str(t.lemma_) for t in nlp(row["raw_4"]) if t.pos_ in included_tags]))
                testDataset["sentid"].append(row["sentids"])
                testDataset["sentid"].append(row["sentids_1"])
                testDataset["sentid"].append(row["sentids_2"])
                testDataset["sentid"].append(row["sentids_3"])
                testDataset["sentid"].append(row["sentids_4"])


    trainDataset = pd.DataFrame.from_dict(trainDataset)
    valDataset = pd.DataFrame.from_dict(valDataset)
    testDataset = pd.DataFrame.from_dict(testDataset)
    trainDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "traindf.csv"))
    valDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "valdf.csv"))
    testDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "testdf.csv"))
    class2id, id2class = utils.encodeDatasetLabels("NWPU-Captions", trainDataset=trainDataset, validationDataset=valDataset, testDataset=testDataset)
    encodings = {"class2id": class2id, "id2class": id2class}

    merged_df = pd.concat([trainDataset, valDataset, testDataset], ignore_index=True)
    class2id = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "r"))["class2id"]

    sentid_counter = merged_df["sentid"].max() + 1
    extended_captions = {"image": [], "class": [], "caption": [], "filtered_caption": [], "sentid": [], "class_caption": [], "img_id": []}
    for image in trainDataset["image"].unique():
        filtered_df = trainDataset[trainDataset["image"] == image]
        for i0, i1 in combinations(range(3), 2):
            extended_captions["image"].append(image)
            extended_captions["img_id"].append(filtered_df.iloc[i0]["img_id"])
            extended_captions["class"].append(filtered_df.iloc[i0]["class"])
            extended_captions["class_caption"].append(filtered_df.iloc[i0]["class_caption"])
            extended_captions["caption"].append(f"{filtered_df.iloc[i0]['caption']} {filtered_df.iloc[i1]['caption']}")
            extended_captions["filtered_caption"].append(f"{filtered_df.iloc[i0]['filtered_caption']} {filtered_df.iloc[i1]['filtered_caption']}")
            extended_captions["sentid"].append(sentid_counter)
            sentid_counter += 1
    trainDataset = pd.concat([trainDataset, pd.DataFrame().from_dict(extended_captions)], ignore_index=True)
    
    extended_captions = {"image": [], "class": [], "caption": [], "filtered_caption": [], "sentid": [], "class_caption": [], "img_id": []}
    for image in valDataset["image"].unique():
        filtered_df = valDataset[valDataset["image"] == image]
        for i0, i1 in combinations(range(3), 2):
            extended_captions["image"].append(image)
            extended_captions["img_id"].append(filtered_df.iloc[i0]["img_id"])
            extended_captions["class"].append(filtered_df.iloc[i0]["class"])
            extended_captions["class_caption"].append(filtered_df.iloc[i0]["class_caption"])
            extended_captions["caption"].append(f"{filtered_df.iloc[i0]['caption']} {filtered_df.iloc[i1]['caption']}")
            extended_captions["filtered_caption"].append(f"{filtered_df.iloc[i0]['filtered_caption']} {filtered_df.iloc[i1]['filtered_caption']}")
            extended_captions["sentid"].append(sentid_counter)
            sentid_counter += 1
    valDataset = pd.concat([valDataset, pd.DataFrame().from_dict(extended_captions)], ignore_index=True)

    extended_captions = {"image": [], "class": [], "caption": [], "filtered_caption": [], "sentid": [], "class_caption": [], "img_id": []}
    for image in testDataset["image"].unique():
        filtered_df = testDataset[testDataset["image"] == image]
        for i0, i1 in combinations(range(3), 2):
            extended_captions["image"].append(image)
            extended_captions["img_id"].append(filtered_df.iloc[i0]["img_id"])
            extended_captions["class"].append(filtered_df.iloc[i0]["class"])
            extended_captions["class_caption"].append(filtered_df.iloc[i0]["class_caption"])
            extended_captions["caption"].append(f"{filtered_df.iloc[i0]['caption']} {filtered_df.iloc[i1]['caption']}")
            extended_captions["filtered_caption"].append(f"{filtered_df.iloc[i0]['filtered_caption']} {filtered_df.iloc[i1]['filtered_caption']}")
            extended_captions["sentid"].append(sentid_counter)
            sentid_counter += 1
    testDataset = pd.concat([testDataset, pd.DataFrame().from_dict(extended_captions)], ignore_index=True)

    trainDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "traindf-extended-captions.csv"))
    valDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "valdf-extended-captions.csv"))
    testDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "testdf-extended-captions.csv"))
    with open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "w") as metadata_file:
        json.dump(encodings, metadata_file)