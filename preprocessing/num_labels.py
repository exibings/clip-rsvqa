import pandas as pd
import os
trainDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})

num_labels = {}
merged_df = pd.concat([trainDataset, validationDataset, testDataset])
for category, label in zip(merged_df["category"], merged_df["label"]):
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
print(num_labels)