import H5Datasets
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel
import json 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pretrain = "saved-models/NWPU-Captions/2cjgamc5-NWPU-Captions:blr0e+00-plr1e-06-wf10-adamw/cp-1"
model = CLIPModel.from_pretrained(model_pretrain)
model.eval()
model.to(device)
encodings = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "r"))
dataset = H5Datasets.NwpuCaptionsKNN("validation")
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

accuracy = {
    "k=1": 0,
    "k=3": 0,
    "k=5": 0,
    "k=10": 0
}

progress_bar = tqdm(range(len(dataset_loader)), desc="KNN Evaluation")
for batch in dataset_loader:
    model_input = {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "input_ids": torch.as_tensor(dataset.classes_input_ids).to(device, non_blocking=True),
        "attention_mask": torch.as_tensor(dataset.classes_attention_mask).to(device, non_blocking=True),
        "return_dict": True
    }
    outputs = model(**model_input)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    probs = probs.detach().cpu().tolist()[0]
    predictions = [(encodings["id2class"][str(idx)], probs[idx]) for idx in range(len(encodings["id2class"]))]
    predictions.sort(key=lambda prediction_pair: prediction_pair[1], reverse=True)
    knn = [1, 3, 5, 10]
    for k in knn:
        pred_captions_k = [prediction[0] for prediction in predictions][:k]
        #print(k,":", pred_captions_k)
        if encodings["id2class"][str(batch["class"].item())] in pred_captions_k:
            accuracy["k=" + str(k)] += 1
    progress_bar.update(1)
progress_bar.close()

print(f"KNN performance for {model_pretrain}:")
print("\tk=1:", accuracy["k=1"]/len(dataset))
print("\tk=3:", accuracy["k=3"]/len(dataset))
print("\tk=5:", accuracy["k=5"]/len(dataset))
print("\tk=10:", accuracy["k=10"]/len(dataset))