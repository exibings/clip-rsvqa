import H5Datasets
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel
import json 
import evaluate
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pretrain = "saved-models/NWPU-Captions/2ewtg0a1-NWPU-Captions:blr1e-08-plr1e-05-wf5-adamw/cp-1"
model = CLIPModel.from_pretrained(model_pretrain)
model.eval()
model.to(device)
#train_dataset = H5Datasets.RsvqaZeroDataset("RSVQA-LR", "train")
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
#validation_dataset = H5Datasets.RsvqaZeroDataset("RSVQA-LR", "validation")
#validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
test_dataset = H5Datasets.RsvqaZeroDataset("RSVQA-LR", "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.num_labels["total"], shuffle=False, num_workers=4, pin_memory=True)

metrics = {"overall": {"accuracy": evaluate.load("accuracy")}}
for question_type in test_dataset.categories:
    metrics[question_type] = {"accuracy": evaluate.load("accuracy")}

progress_bar = tqdm(range(len(test_loader)), desc="Zero Shot Classification")
for batch in test_loader:
    model_input = {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "return_dict": True
    }
    outputs = model(**model_input)
    probs = outputs.logits_per_image.softmax(dim=1)
    probs = probs.detach().cpu().tolist()[0]
    predictions = [{"prompt": prompt, "probability": prob, "correct_prompt": truth.item(), "category": category} for (prompt, prob, truth, category) in zip(batch["prompt"], probs, batch["correct_prompt"], batch["category"])]
    prediction = max(predictions, key=lambda prediction: prediction["probability"])
    metrics[prediction["category"]]["accuracy"].add(prediction=prediction["correct_prompt"], reference=1)
    metrics["overall"]["accuracy"].add(prediction=prediction["correct_prompt"], reference=1)
    progress_bar.update(1)
progress_bar.close()

total_train_accuracy = 0
for question_type in metrics:
    if question_type != "overall":
        accuracy = metrics[question_type]["accuracy"].compute()["accuracy"]
        total_train_accuracy += accuracy
        print({"test/" + question_type + " accuracy": accuracy})
print({"test/average accuracy": total_train_accuracy / (len(metrics)-1)})
print({"test/overall accuracy": metrics["overall"]["accuracy"].compute()["accuracy"]})
