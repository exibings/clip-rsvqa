import os
import json
import pandas as pd
import spacy
from itertools import combinations
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from statistics import stdev

nlp = spacy.load('en_core_web_md')

file_name = "nwpu_captions.h5"
dataset_folder =  os.path.join("datasets", "NWPU-Captions")
encodings = json.load(open(os.path.join(dataset_folder, "nwpu_captions_metadata.json"), "r"))
class2id = encodings["class2id"]
id2class = encodings["id2class"]


dataset = pd.read_csv(os.path.join(dataset_folder, "traindf.csv"), index_col="sentid")
dataset = dataset.drop([14283])

def compute_similarity(caption1, caption2):
    #print(caption1, "|", caption2)
    return nlp(caption1).similarity(nlp(caption2))

def compute_similarities(captions: list[str]) -> float:
    total_sim = 0
    captions = combinations(captions, 2)
    n_pairs = 0
    for pair in captions:
        n_pairs += 1
        total_sim += compute_similarity(pair[0], pair[1])
    return total_sim / n_pairs

def get_captions_by_image(image: str) -> list[str]:
    sub_df = dataset[dataset["image"] == image]
    return sub_df["filtered_caption"].tolist()

def get_captions_by_class(_class: str) -> list[str]:
    sub_df = dataset[dataset["class"] == _class]
    return sub_df["filtered_caption"].tolist()

def get_caption(sentid: int) -> str:
    return dataset.loc[sentid, "filtered_caption"]

def generate_pairs(fixed_caption: str, to_be_combined_captions: list[str], classes: list[str]) -> list:
    result = {}
    for _class in set(classes):
        result[_class] = []
    for caption, _class in zip(to_be_combined_captions, classes):
        result[_class].append((fixed_caption, caption))
    return result

def fixed_class_study(n_pairs:int, _class: str) -> dict[str, float]:
    filtered_dataset = dataset[dataset["class"] == _class]
    starting_sample = filtered_dataset.sample()
    to_be_combined_samples = dataset.groupby("class").sample(n=n_pairs).to_dict(orient="list")
    combined_captions = generate_pairs(starting_sample["filtered_caption"].item(), to_be_combined_samples["filtered_caption"], to_be_combined_samples["class"])
    average = {}
    for _class1 in combined_captions:
        average[_class1] = 0
        for caption in combined_captions[_class1]:
            average[_class1] += compute_similarity(caption[0], caption[1])
        average[_class1] = average[_class1] / len(combined_captions[_class1])
    
    sorted_average = {k: v for k, v in sorted(average.items(), key=lambda item: item[1])}
    result = {"fixed_sample": {
        "class":starting_sample["class"].tolist()[0],
        "filtered caption":starting_sample["filtered_caption"].tolist()[0]
        }}
    
    result.update(sorted_average)
    with open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "fixed-class-similarity", "study-" + str(_class) + ".json"), "w") as similarity_file:
        json.dump(result, similarity_file, indent=4, separators=(',', ': '))
    return result

def image_caption_similarity_study() -> dict[str, float]:
    result = {}
    for image in dataset["image"].tolist():
        result[image] = compute_similarities(get_captions_by_image(image))
    with open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "image-captions-similarity.json"), "w") as similarity_file:
        json.dump(result, similarity_file, indent=4, separators=(',', ': '))
    return result

def same_class_study_per_class(n_pairs: int, _class: str) -> None:
    filtered_dataset = dataset[dataset["class"] == _class]
    
    to_be_combined_captions = filtered_dataset.groupby("class").sample(n=n_pairs*2)["filtered_caption"].tolist()
    combined_captions = [tuple(to_be_combined_captions[n:n+2]) for n in range(0, len(to_be_combined_captions), 2)]
    avg_similarity = 0
    for caption_pair in combined_captions:
        try:
            avg_similarity += compute_similarity(caption_pair[0], caption_pair[1])
        except:
            print("class:", _class, "faulty caption pair:", caption_pair)
    return avg_similarity / len(combined_captions)

def same_class_study(n_pairs: int, ) -> pd.DataFrame:
    progress_bar = tqdm(range(n_pairs), desc="Same class study")
    similarity_values = []
    for _ in range(n_pairs):
        sample1 = dataset.sample()
        filtered_dataset = dataset[(dataset["class"] == sample1["class"].item()) & (dataset["image"] != sample1["image"].item())]
        sample2 = filtered_dataset.sample()
        try:
            similarity_values.append(compute_similarity(sample1["filtered_caption"].item(), sample2["filtered_caption"].item()))
        except:
            print("faulty caption:", sample1["filtered_caption"].item(), sample2["filtered_caption"].item())
        progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(similarity_values, columns=["similarity"]).sort_values(by="similarity")

def diff_class_study(n_pairs: int) -> pd.DataFrame:
    progress_bar = tqdm(range(n_pairs), desc="Different class study")
    similarity_values = []
    for _ in range(n_pairs):
        sample1 = dataset.sample()
        filtered_dataset = dataset[dataset["class"] != sample1["class"].item()]
        sample2 = filtered_dataset.sample()
        try:
            similarity_values.append(compute_similarity(sample1["filtered_caption"].item(), sample2["filtered_caption"].item()))
        except:
            print("faulty caption:", sample1["filtered_caption"].item(), sample2["filtered_caption"].item())
        progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(similarity_values, columns=["similarity"]).sort_values(by="similarity")

progress_bar = tqdm(range(len(class2id)), desc="Fixed class study")
for _class in class2id:
    fixed_class_study(n_pairs=1000, _class=_class)
    progress_bar.update(1)
progress_bar.close()
for i in range(3):
    same_class_study_results = {}
    progress_bar = tqdm(range(len(class2id)), desc="Same class study per class")
    for _class in class2id:
        same_class_study_results[_class] = same_class_study_per_class(n_pairs=1000, _class = _class)
        progress_bar.update(1)
    sorted_average = {k: v for k, v in sorted(same_class_study_results.items(), key=lambda item: item[1])}
    progress_bar.close()
    with open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-" + str(i+1) + ".json"), "w") as similarity_file:
        json.dump(sorted_average, similarity_file, indent=4, separators=(',', ': '))
    diff_similarities = diff_class_study(n_pairs=45000)
    diff_similarities.to_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "diff-class-similarity","diff-class-study-" + str(i+1) + ".csv"))
    same_similarities = same_class_study(n_pairs=45000)
    same_similarities.to_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-" + str(i+1) + ".csv"))

#Generate diff-class-similarity histogram
df1 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "diff-class-similarity","diff-class-study-1.csv"))
df2 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "diff-class-similarity","diff-class-study-2.csv"))
df3 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "diff-class-similarity","diff-class-study-3.csv"))
print("df1 similarity 75th percentile:", df1.similarity.quantile(0.75))
print("df2 similarity 75th percentile:", df2.similarity.quantile(0.75))
print("df3 similarity 75th percentile:", df3.similarity.quantile(0.75))
print("different class mean 75th percentile:", (df1.similarity.quantile(0.75) + df2.similarity.quantile(0.75) + df3.similarity.quantile(0.75))/3, "±",
                                                stdev([df1.similarity.quantile(0.75), df2.similarity.quantile(0.75), df3.similarity.quantile(0.75)]))
plt.figure()
plt.hist(df1["similarity"], bins=50, alpha=0.5)
plt.hist(df2["similarity"], bins=50, alpha=0.5)
plt.hist(df3["similarity"], bins=50, alpha=0.5)
plt.xlabel("Similarity")
plt.xlim(0, 1)
plt.ylabel("Count")
plt.title("NWPU-Captions Similarity Distribution between Classes")
plt.savefig(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "diff-class-similarity","diff-class-similarity.svg"))

# Generate same-class-similarity histogram
df4 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-1.csv"))
df5 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-2.csv"))
df6 = pd.read_csv(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-3.csv"))
print("df4 similarity 25th percentile:", df4.similarity.quantile(0.25))
print("df5 similarity 25th percentile:", df5.similarity.quantile(0.25))
print("df6 similarity 25th percentile:", df6.similarity.quantile(0.25))
print("same class mean 25th percentile:", (df4.similarity.quantile(0.25) + df5.similarity.quantile(0.25) + df6.similarity.quantile(0.25))/3, "±",
                                                stdev([df4.similarity.quantile(0.25), df5.similarity.quantile(0.25), df6.similarity.quantile(0.25)]))
plt.figure()
plt.hist(df4["similarity"], bins=50, alpha=0.5)
plt.hist(df5["similarity"], bins=50, alpha=0.5)
plt.hist(df6["similarity"], bins=50, alpha=0.5)
plt.xlabel("Similarity")
plt.xlim(0, 1)
plt.ylabel("Count")
plt.title("NWPU-Captions Similarity Distribution same Classes")
plt.savefig(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-similarity.svg"))

# Generate merged similarity histograms for comparison
merged_diff_class = pd.concat([df1, df2, df3])
merged_same_class = pd.concat([df4, df5, df6])
plt.figure()
plt.hist(merged_diff_class["similarity"], bins=50, alpha=0.5, label="Different Class")
plt.hist(merged_same_class["similarity"], bins=50, alpha=0.5, label="Same Class")
plt.xlabel("Similarity")
plt.xlim(0, 1)
plt.ylabel("Count")
plt.title("NWPU-Captions Similarity Distribution Comparison")
plt.legend()
plt.savefig(os.path.join("datasets", "NWPU-Captions", "similarity-studies","class-similarity-comparison.svg"))

# Generate same-class-similarity per class histogram
similarities1 = json.load(open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-1.json"), "r"))
similarities2 = json.load(open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-2.json"), "r"))
similarities3 = json.load(open(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-study-3.json"), "r"))
same_class_mean = {}
error = {}
for key in similarities1:
    error[key] = stdev([similarities1[key], similarities2[key], similarities3[key]])
    same_class_mean[key] = (similarities1[key] + similarities2[key] + similarities3[key]) / 3
df7 = pd.DataFrame.from_dict(similarities1, orient="index", columns=["similarity"])
df8 = pd.DataFrame.from_dict(similarities2, orient="index", columns=["similarity"])
df9 = pd.DataFrame.from_dict(similarities3, orient="index", columns=["similarity"])
fig, ax = plt.subplots()
fig.set_size_inches(14, 8)
ax.barh(range(len(same_class_mean)), same_class_mean.values(), xerr=error.values(), align="center")
ax.set_yticks(range(len(same_class_mean)), labels=same_class_mean.keys())
ax.set_xlabel('Similarity')
ax.set_title('NWPU-Captions Mean Caption Similarity per Class')
plt.savefig(os.path.join("datasets", "NWPU-Captions", "similarity-studies", "same-class-similarity","same-class-similarity-per-class.svg"))



