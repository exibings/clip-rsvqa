import json
import os
from datetime import datetime
import H5Datasets
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel
import wandb
import math


class PreTrainer:
    def __init__(self, device: torch.device):
        self.run_config = {
                "batch size": 90, # n_classes * 2
                "limit epochs": 1,
                "initial learning rate": 1e-8,
                "peak learning rate": 1e-5,
                "dataset": "NWPU-Captions",
                "logging steps": 20,
                "learning rate warmup fraction": 0.05
        }
        self.run_name = f"{self.run_config['dataset']:s}:blr{self.run_config['initial learning rate']:.0e}-plr{self.run_config['peak learning rate']:.0e}-wf{int(self.run_config['learning rate warmup fraction']*100):d}-adamw"
        wandb.init(project="CLIPxRSVQA", job_type="pre-train", name=self.run_name, config=self.run_config)
        self.run_name = f"{wandb.run.id:s}-{self.run_name:s}"
        wandb.run.name = self.run_name
        self.run_folder = os.path.join("saved-models", self.run_config["dataset"], self.run_name)
        os.makedirs(self.run_folder)
        self.model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        self.logging_steps = self.run_config["logging steps"]
        self.device = device
        self.batch_size = self.run_config["batch size"]
        self.limit_epochs = self.run_config["limit epochs"]
        self.lr = self.run_config["initial learning rate"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))
        self.dataset_name = self.run_config["dataset"]
        self.train_dataset = H5Datasets.NwpuCaptionsDataset("nwpu_captions_bigger.h5", "train", augment_images=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.validation_dataset = H5Datasets.NwpuCaptionsDataset("nwpu_captions_bigger.h5", "validation", augment_images=True)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.test_dataset = H5Datasets.NwpuCaptionsDataset("nwpu_captions_bigger.h5", "test", augment_images=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.metadata = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_metadata.json"), "r"))
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                            step_size_up=math.floor(self.run_config["learning rate warmup fraction"]*(len(self.train_loader))), 
                                                            step_size_down=math.floor((1-self.run_config["learning rate warmup fraction"])*len(self.train_loader)), 
                                                            base_lr=self.run_config["initial learning rate"], 
                                                            max_lr=self.run_config["peak learning rate"], cycle_momentum=False)
        self.model.to(self.device)  # send model to GPU

    def batchToGPU(self, batch: dict) -> dict:
        """
        Sends batch to GPU.

        Args:
            `batch` (dict): batch to be sent to GPU.

        Returns:
            A dictionary with keys: `input_ids`, `attention_mask` and `pixel_values` ready to be fed to the CLIP Model.
        """
        processed_batch = {"return_loss": True}
        processed_batch["input_ids"] = batch["input_ids"].to(self.device, non_blocking=True) 
        processed_batch["attention_mask"] = batch["attention_mask"].to(self.device, non_blocking=True) 
        processed_batch["pixel_values"] = batch["pixel_values"].to(self.device, non_blocking=True)
        return processed_batch

    def saveModel(self, current_epoch: int) -> None:
        """
        Updates the currently saved models to only keep the current best model
        
        Args:
            `current_epoch` (int): epoch of the model that it is being saved.
            `validation_losses` dict[int, float]: map of <#epoch, validation loss>.
        """
        self.checkpoint_path = os.path.join(self.run_folder, f"cp-{current_epoch:d}")
        wandb.run.summary["best model"] = self.checkpoint_path
        self.model.save_pretrained(self.checkpoint_path)

    def train(self) -> None:
        """
        Train function that will iterate over the training dataset, back-propagate the loss and step the optimizer. 
        During train the learning rate will follow a Cyclic strategy, increasing from its base level to its peak value. 
        After reaching the peak the learning rate will start decreasing until the end of the train loop.
        The main train loop is finished when the current epoch exceeds `self.limit_epochs`.
        Saves the trained model after finishing the main loop. 
        """
        epoch_count = 1
        
        while epoch_count <= self.limit_epochs:
            progress_bar = tqdm(range(len(self.train_loader)), desc=datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - epoch "+ str(epoch_count))
            # train loop
            running_loss = 0.0
            step = 0
            for batch in self.train_loader:
                # train step
                batch = self.batchToGPU(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model(**batch)["loss"]
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                # /train step
                if step % self.logging_steps == 0 and step > 0:
                    wandb.log({"train/clip loss": running_loss/self.logging_steps, "learning rate": self.lr_scheduler.optimizer.param_groups[0]["lr"]})
                    running_loss = 0.0
                step += 1
                self.lr_scheduler.step() # update learning rate
                progress_bar.update(1)
            progress_bar.close()
            # finished epoch
            self.saveModel(epoch_count)
            epoch_count += 1

    def test(self) -> None:
        """
        Computes the average loss of the model over the test dataset and logs the running loss after `self.logging_steps` steps. 
        At the end updates the run summary with the average test loss.
        """
        total_loss = 0.0
        running_loss = 0.0
        step = 0
        self.model = CLIPModel.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)  # send model to GPU
        self.model.eval()
        progress_bar = tqdm(range(len(self.test_loader)), desc="Computing metrics for test dataset")
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.batchToGPU(batch)
                loss = self.model(**batch)["loss"].item()
                running_loss += loss
                total_loss += loss
                if step % self.logging_steps == 0 and step > 0:
                    wandb.log({"test/clip loss": running_loss/self.logging_steps})
                    running_loss = 0.0
                step += 1
                progress_bar.update(1)
        progress_bar.close()
        wandb.run.summary["test/average clip loss"] = total_loss/len(self.test_loader)
   
    def run(self) -> None:
        """
        Run loop. Trains the model and then evaluates it.
        """
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- started run")     
        self.train()
        self.test()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- finished run")     
