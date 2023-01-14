import H5Datasets
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel
from transformers import CLIPTokenizer
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pretrain = "saved-models/NWPU-Captions/137n9s6h-NWPU-Captions:blr1e-08-plr1e-05-wf25-adamw/cp-1"
model = CLIPModel.from_pretrained(model_pretrain)
model.eval()
model.to(device)
print(f"CLIP Model pretrain loaded from {model_pretrain}")
captions_dataset = H5Datasets.NwpuCaptionsSentences("nwpu_captions.h5", "validation", False)
captions_loader = torch.utils.data.DataLoader(captions_dataset, batch_size=200, shuffle=False, num_workers=4, pin_memory=True)
image_dataset = H5Datasets.NwpuCaptionsImages("nwpu_captions.h5", "validation", False)
image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

index = faiss.IndexFlatL2(512)
captions = []
progress_bar = tqdm(range(len(captions_loader)), desc="Getting text embeddings and adding them to index")
for batch in captions_loader:
    output = model.get_text_features(input_ids=batch["input_ids"].to(device, non_blocking=True), attention_mask=batch["attention_mask"].to(device, non_blocking=True), return_dict=True).cpu().detach().numpy()
    for caption in batch["caption"]:
        captions.append(caption)
    index.add(output)
    progress_bar.update(1)
progress_bar.close()

knn = [1, 3, 5, 10]
accuracy = {"k=1": 0, "k=3": 0, "k=5": 0, "k=10":0}

progress_bar = tqdm(range(len(image_loader)), desc="Searching index by image")
for batch in image_loader:
    #print("Search with image:", batch["image"][0])
    #print("Correct captions:", image_dataset.image_to_captions[batch["image"][0]])
    output = model.get_image_features(pixel_values=batch["pixel_values"].to(device, non_blocking=True), return_dict=True).cpu().detach().numpy()
    _, I = index.search(output, 10)
    I = I.tolist()
    pred_captions = [captions[idx] for idx in I[0]]
    for k in knn:
        pred_captions_k = pred_captions[:k]
        #print(k,":", pred_captions_k)
        for caption in pred_captions_k:
            if caption in image_dataset.image_to_captions[batch["image"][0]]:
                accuracy["k=" + str(k)] += 1
    #print(accuracy)
    progress_bar.update(1)
progress_bar.close()

print(f"Accuracies for {model_pretrain}:")
print("\tk=1:", accuracy["k=1"]/len(captions_dataset))
print("\tk=3:", accuracy["k=3"]/len(captions_dataset))
print("\tk=5:", accuracy["k=5"]/len(captions_dataset))
print("\tk=10:", accuracy["k=10"]/len(captions_dataset))
