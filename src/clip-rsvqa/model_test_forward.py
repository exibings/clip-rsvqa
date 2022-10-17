import torch
import os
from H5Dataset import H5Dataset
import json 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "RSVQA-LR"
model_name = "baseline"
if model_name == "baseline":
    from Models.Baseline import CLIPxRSVQA
elif model_name == "patching":
    from Models.Patching import CLIPxRSVQA
train_dataset = H5Dataset(os.path.join("datasets", dataset_name, "rsvqa_lr.h5"), "train", model_name)
# load label encodings
id2label = json.load(open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_id2label.json"), "r"))
label2id = json.load(open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_label2id.json"), "r"))

model = CLIPxRSVQA(num_labels=len(label2id))
model.to(device)  # send model to GPU

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=80,
                                             shuffle=False, pin_memory=True, num_workers=4)

#print("model.name=", model.name)
batch = next(iter(train_dataset_loader))
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
logits = model(input_ids=batch["input_ids"],
            attention_mask = batch["attention_mask"],
            pixel_values=batch["pixel_values"])
predictions =  torch.argmax(logits, dim=-1)
print(predictions)

