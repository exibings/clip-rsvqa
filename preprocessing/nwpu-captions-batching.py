import os
import json
import pandas as pd
from tqdm.auto import tqdm
from random import uniform
from itertools import combinations
import spacy

nlp = spacy.load('en_core_web_md')
file_name = "nwpu_captions.h5"
dataset_folder =  os.path.join("datasets", "NWPU-Captions")
encodings = json.load(open(os.path.join(dataset_folder, "nwpu_captions_metadata.json"), "r"))
class2id = encodings["class2id"]
id2class = encodings["id2class"]
dataset = pd.read_csv(os.path.join(dataset_folder, "traindf.csv"))

BATCH_SIZE = len(class2id) * 2
N_BATCHES = len(dataset) * 2 // BATCH_SIZE
LOW_SIMILARITY = 1
HIGH_SIMILARITY = 0

def addSample(batch: dict, sample: dict) -> dict:
    batch["sentid"].append(sample["sentid"][0])
    batch["class"].append(sample["class"][0])
    batch["image"].append(sample["image"][0])
    batch["caption"].append(sample["caption"][0])
    batch["filtered_caption"].append(sample["filtered_caption"][0])
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

batch_counter = 0
progress_bar = tqdm(range(N_BATCHES))
batches = []
while batch_counter < N_BATCHES:
    batch = {"sentid": [], "class": [], "image":[], "caption": [], "filtered_caption": []}
    starting_sample = dataset.sample().to_dict(orient="list")
    batch = addSample(batch, starting_sample)
    sample_counter = 1
    while sample_counter < BATCH_SIZE:
        sample_found = False
        if uniform(0, 1) >= 0.5:
            # new sample from a class already in the batch - needs to have LOW similarity with the captions from that same class in the batch
            while not sample_found:
                # new sample from a class already in the batch
                filtered_dataset = dataset[dataset["class"].isin(batch["class"])]
                sample = filtered_dataset.sample().to_dict(orient="list")
                caption_pairs = generate_pairs(sample["filtered_caption"][0], get_filtered_captions_by_class(batch, sample["class"][0]))
                # TODO adicionar patience para os thresholds - acho que é só fazer um isto dentro de um loop e contar as iterações até ter sample found
                if compute_avg_similarity(caption_pairs) < LOW_SIMILARITY:
                    # low average similarity with all the captions from the same class
                    batch = addSample(batch, sample)
                    sample_found = True


        else:
            # new sample from a class NOT in the batch - needs to have HIGH similarity with one of the captions from the batch
            while not sample_found:
                # new sample from a class NOT in the batch
                filtered_dataset = dataset[~dataset["class"].isin(batch["class"])]
                if len(filtered_dataset) == 0:
                    # all the classes are in the batch. instead try to find samples from images that are not in the batch
                    print("all classes in the batch!")
                    filtered_dataset = dataset[~dataset["image"].isin(batch["image"])]
                # TODO adicionar patience para os thresholds - acho que é só fazer um isto dentro de um loop e contar as iterações até ter sample found
                sample = filtered_dataset.sample().to_dict(orient="list")
                caption_pairs = generate_pairs(sample["filtered_caption"][0], [caption for caption in batch["filtered_caption"]])
                computed_similarities = compute_similarities(caption_pairs)
                if any(similarity > HIGH_SIMILARITY for similarity in computed_similarities):
                    # high similarity with one the captions from the batch
                    batch = addSample(batch, sample)
                    sample_found = True
        sample_counter += 1
        print("samples:", sample_counter)

    batches.append(batch)
    batch_counter += 1
    progress_bar.update(1)


print("n batches:", len(batches))

