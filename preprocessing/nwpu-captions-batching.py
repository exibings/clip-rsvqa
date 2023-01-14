import os
import json
import pandas as pd
from tqdm.auto import tqdm
from random import uniform, sample
import spacy
import h5py
import utils

dataset_folder =  os.path.join("datasets", "NWPU-Captions")
encodings = json.load(open(os.path.join(dataset_folder, "nwpu_captions_metadata.json"), "r"))
class2id = encodings["class2id"]
id2class = encodings["id2class"]

nlp = spacy.load('en_core_web_md')
BATCH_SIZE = len(class2id) * 2
SAME_CLASS_THRESHOLD = 0.665
DIFF_CLASS_THRESHOLD = 0.665

def addSample(batch: dict, sample: dict) -> dict:
    batch["sentid"].append(sample["sentid"].item())
    batch["class"].append(sample["class"].item())
    batch["image"].append(sample["image"].item())
    batch["caption"].append(sample["caption"].item())
    batch["filtered_caption"].append(sample["filtered_caption"].item())
    return batch

def compute_similarity(caption1, caption2):
    #print(caption1, "|", caption2)
    return nlp(caption1).similarity(nlp(caption2))

def generate_pairs(fixed_caption: str, to_be_combined_captions: list[str]) -> list[tuple[str, str]]:
    return [(fixed_caption, caption) for caption in to_be_combined_captions]

def compute_avg_similarity(captions: list[str]) -> float:
    total_sim = 0
    n_pairs = 0
    for pair in captions:
        n_pairs += 1
        total_sim += compute_similarity(pair[0], pair[1])
    return total_sim / n_pairs

def compute_similarities(captions: list[str]) -> list:
    return [compute_similarity(pair[0], pair[1]) for pair in captions]

def get_filtered_captions_by_class(batch: dict, fixed_class:str) -> list[str]:
    result = []
    for _class, filtered_caption in zip(batch["class"], batch["filtered_caption"]):
        if _class == fixed_class:
            result.append(filtered_caption)
    return result

def batching_alt(dataset: pd.DataFrame) -> dict:
    batch = {"sentid": [], "class": [], "image":[], "caption": [], "filtered_caption": []}
    starting_sample = dataset.sample()
    batch = addSample(batch, starting_sample)
    sample_counter = 1
    while sample_counter < BATCH_SIZE:
        sample_found = False
        if uniform(0, 1) >= 0.5:
            # new sample from a class already in the batch - needs to have LOW similarity with the captions from that same class in the batch
            threshold = SAME_CLASS_THRESHOLD
            patience = 0
            while not sample_found:
                # new sample from a class already in the batch but no repeat
                filtered_dataset = dataset[(dataset["class"].isin(batch["class"])) & ~(dataset["sentid"].isin(batch["sentid"]))]
                sample = filtered_dataset.sample()
                caption_pairs = generate_pairs(sample["filtered_caption"].item(), get_filtered_captions_by_class(batch, sample["class"].item()))
                if compute_avg_similarity(caption_pairs) < threshold:
                    # low average similarity with all the captions from the same class
                    batch = addSample(batch, sample)
                    sample_found = True
                    patience = 0
                else:
                    patience += 1
                if patience == 5:
                    threshold += 0.05
                    patience = 0
        else:
            # new sample from a class NOT in the batch - needs to have HIGH similarity with one of the captions from the batch
            threshold = DIFF_CLASS_THRESHOLD
            patience = 0
            while not sample_found:
                # new sample from a class NOT in the batch
                filtered_dataset = dataset[~dataset["class"].isin(batch["class"])]
                if len(filtered_dataset) == 0:
                    # all the classes are in the batch. instead try to find samples from images that are not in the batch
                    filtered_dataset = dataset[~dataset["image"].isin(batch["image"])]
                sample = filtered_dataset.sample()
                caption_pairs = generate_pairs(sample["filtered_caption"].item(), [caption for caption in batch["filtered_caption"]])
                computed_similarities = compute_similarities(caption_pairs)
                if any(similarity > threshold for similarity in computed_similarities):
                    # high similarity with one the captions from the batch
                    batch = addSample(batch, sample)
                    sample_found = True
                    patience = 0
                else:
                    patience += 1
                if patience == 5:
                    threshold -= 0.05
                    patience = 0
        sample_counter += 1
    return batch

def batching2(dataset: pd.DataFrame) -> dict:
    batch = {"sentid": [], "class": [], "image":[], "caption": [], "filtered_caption": []}
    for _class in sample(sorted(class2id), len(class2id)):
        # new sample from a new class - needs to have HIGH similarity with any of the other samples already in the batch
        filtered_dataset = dataset[dataset["class"] == _class]
        sample_found = False
        threshold = DIFF_CLASS_THRESHOLD
        patience = 0
        while not sample_found:
            _sample = filtered_dataset.sample()
            caption_pairs = generate_pairs(_sample["filtered_caption"].item(), [caption for caption in batch["filtered_caption"]])
            computed_similarities = compute_similarities(caption_pairs)
            if any(similarity > threshold for similarity in computed_similarities) or len(computed_similarities) == 0:
                # high similarity with one the captions from the batch
                batch = addSample(batch, _sample)
                sample_found = True
            else:
                patience += 1
            if patience == 5:
                threshold -= 0.05
                patience = 0
        
        # second sample from the new class - needs to have LOW similarity with previous sample from the same class
        filtered_dataset = filtered_dataset[filtered_dataset["image"] != _sample["image"].item()]
        sample_found = False
        threshold = SAME_CLASS_THRESHOLD
        patience = 0
        while not sample_found:
            _sample = filtered_dataset.sample()
            caption_pairs = generate_pairs(_sample["filtered_caption"].item(), get_filtered_captions_by_class(batch, _sample["class"].item()))
            if compute_avg_similarity(caption_pairs) < threshold:
                # low average similarity with all the captions from the same class
                batch = addSample(batch, _sample)
                sample_found = True
                patience = 0
            else:
                patience += 1
            if patience == 5:
                threshold += 0.05
                patience = 0
    return batch

def generate_batches(split, dataset):
    n_batches = 2 * (len(dataset) * 2 // BATCH_SIZE)
    batch_counter = 0
    progress_bar = tqdm(range(n_batches), desc="Generating " + split + " batches")
    batches = []
    while batch_counter < n_batches:
        batches.append(batching2(dataset))
        batch_counter += 1
        progress_bar.update(1)
    progress_bar.close()
    flat_batches_2 = {"sentid": [], "class": [], "image": [], "caption": [], "filtered_caption": []}
    for batch in batches:
        for key in batch:
            flat_batches_2[key] += batch[key]
    batches = pd.DataFrame.from_dict(flat_batches_2)
    batches.to_csv(os.path.join("datasets", "NWPU-Captions", split + "_batched_bigger" + ".csv"), index=False)


train_dataset = pd.read_csv(os.path.join(dataset_folder, "traindf.csv"), index_col="sentid")
train_dataset["sentid"] = train_dataset.index
train_dataset = train_dataset.drop([14283])
val_dataset = pd.read_csv(os.path.join(dataset_folder, "valdf.csv"), index_col="sentid")
val_dataset["sentid"] = val_dataset.index
test_dataset = pd.read_csv(os.path.join(dataset_folder, "testdf.csv"), index_col="sentid")
test_dataset["sentid"] = test_dataset.index
test_dataset = test_dataset.drop([52202])

generate_batches("train", train_dataset)
generate_batches("val", val_dataset)
generate_batches("test", test_dataset)
train_dataset = pd.read_csv(os.path.join(dataset_folder, "train_batched_bigger.csv"))
val_dataset = pd.read_csv(os.path.join(dataset_folder, "val_batched_bigger.csv"))
test_dataset = pd.read_csv(os.path.join(dataset_folder, "test_batched_bigger.csv"))

with h5py.File(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_batched_bigger.h5"), "w") as hfile:
    utils.createDatasetSplit("NWPU-Captions", hfile, "train", train_dataset, label2id_encodings=class2id)
    del train_dataset
    utils.createDatasetSplit("NWPU-Captions", hfile, "validation", val_dataset, label2id_encodings=class2id)
    del val_dataset
    utils.createDatasetSplit("NWPU-Captions", hfile, "test", test_dataset, label2id_encodings=class2id)
    del test_dataset