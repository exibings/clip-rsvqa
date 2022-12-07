import os
import json
import pandas as pd
import h5py
import utils
import spacy


nlp = spacy.load('en_core_web_md')
included_tags = {"NOUN", "ADJ", "NUM"}

trainDataset = {"class": [], "image": [], "caption": [], "filtered_caption": [], "sentid": []}
testDataset = {"class": [], "image": [], "caption": [],  "filtered_caption": [], "sentid": []}
valDataset = {"class": [], "image": [], "caption": [],  "filtered_caption": [], "sentid": []}
with open(os.path.join("datasets", "NWPU-Captions", "dataset_nwpu.json"), "r") as nwpu_file:
    nwpu_data = json.load(nwpu_file)
    
    for label in nwpu_data:
        for row in nwpu_data[label]:
            if row["split"] == "train":
                if row["filename"] == "airplane_111.jpg":
                    continue
                trainDataset["class"].extend(label for _ in range(5))
                trainDataset["image"].extend(row["filename"] for _ in range(5))
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
    
    class2id, id2class = utils.encodeDatasetLabels("NWPU-Captions", trainDataset=trainDataset, validationDataset=valDataset, testDataset=testDataset)
    encodings = {"class2id": class2id, "id2class": id2class}
    with open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "w") as metadata_file:
        json.dump(encodings, metadata_file)
    
    trainDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "traindf.csv"), index=False)
    valDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "valdf.csv"), index=False)
    testDataset.to_csv(os.path.join("datasets", "NWPU-Captions", "testdf.csv"), index=False)
    with h5py.File(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "w") as hfile:
        utils.createDatasetSplit("NWPU-Captions", hfile, "train", trainDataset, label2id_encodings=class2id)
        del trainDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "validation", valDataset, label2id_encodings=class2id)
        del valDataset
        utils.createDatasetSplit("NWPU-Captions", hfile, "test", testDataset, label2id_encodings=class2id)
        del testDataset