import torch
import os
import H5Datasets
import json 
import Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "RSVQAxBEN"
model_name = "baseline"
    
train_dataset = H5Datasets.RsvqaBenDataset("RSVQAxBEN", "train", model_name)
# load label encodings
encodings = json.load(open(os.path.join("datasets", dataset_name, "rsvqaxben_metadata.json"), "r"))
id2label = encodings["id2label"]
label2id = encodings["label2id"]

if model_name == "baseline":
    model = Models.Baseline(num_labels=train_dataset.num_labels["total"], model_aspect_ratio={"n_layers": 1, "n_heads": 32})
    model.to(device)  # send model to GPU
elif model_name == "patching":
    model=Models.Patching(num_labels = train_dataset.num_labels["total"], model_aspect_ratio = {"n_layers": 1, "n_heads": 32})
    model.to(device)  # send model to GPU

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=False, pin_memory=True, num_workers=6)
#print("model.name=", model.name)
batch = next(iter(train_dataset_loader))
print(batch["label"].dtype)
for key in batch:
    if key != "category" and key != "question" and key != "label":
        batch[key] = batch[key].to(device, non_blocking=True)
#print("batch.pixel_values.size()", batch["pixel_values"].size())
#print("batch.pixel_values", batch["pixel_values"])
#print("pixel values size", batch["pixel_values"].size())
#print("patch1 size", batch["pixel_values"][0].size())
#print("patch1", batch["pixel_values"][0])
#print("patch2 size", batch["pixel_values"][1].size())
#print("patch2", batch["pixel_values"][1])
#print("patch3 size", batch["pixel_values"][2].size())
#print("patch3", batch["pixel_values"][2])
#print("patch4 size", batch["pixel_values"][3].size())
#print("patch4", batch["pixel_values"][3])
#print("full image size", batch["pixel_values"][4].size())
#print("full image", batch["pixel_values"][4])
logits = model(input_ids=batch["input_ids"], attention_mask = batch["attention_mask"], pixel_values = batch["pixel_values"])
print("raw model output:", logits)
predictions =  torch.argmax(logits, dim=-1)
for prediction, ground_truth in zip(predictions.tolist(), batch["label"].tolist()):
    print("\tmodel predicted:", id2label[str(prediction)], "\n\tcorrect answer", id2label[str(ground_truth)])

