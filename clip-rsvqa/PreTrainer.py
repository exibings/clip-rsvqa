import json
import os
from datetime import datetime
from typing import Tuple, Union

import H5Datasets
import torch
from tqdm.auto import tqdm
from transformers import CLIPModel
from torchvision.utils import save_image
import wandb


class PreTrainer:
    def __init__(self, batch_size, limit_epochs, patience, lr_patience, device):
        self.run_config = {
                "batch size": 3, # TODO change this to batch_size
                "limit epochs": limit_epochs,
                "patience": patience,
                "initial learning rate": 5e-5,
                "learning rate patience": lr_patience,
                "dataset": "NWPU-Captions",
                "logging steps": 50}

        #wandb.init(project="CLIPxRSVQA", job_type="pre-train", name=self.run_name, config=self.run_config)
        self.run_name = f"bs{self.run_config['batch size']:d}-lr{self.run_config['initial learning rate']:.0e}-lr_p{self.run_config['learning rate patience']:d}-adamw"
        self.run_folder = os.path.join("saved-models", self.run_config["dataset"], self.run_name)
        os.makedirs(self.run_folder)
        # load pre-trained model
        #self.model = CLIPModel.from_pretrained("saved-models/NWPU-Captions/2022-12-03-20h18/bs3-lr5e-05-lr_p10-adamwcp-0")
        self.model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        self.logging_steps = self.run_config["logging steps"]
        self.device = device
        self.batch_size = self.run_config["batch size"]
        self.limit_epochs = self.run_config["limit epochs"]
        self.patience = self.run_config["patience"]
        self.lr = self.run_config["initial learning rate"]
        self.lr_patience = self.run_config["learning rate patience"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience)
        
        self.dataset_name = self.run_config["dataset"]
        self.train_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "train")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.validation_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "validation")
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.test_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "test")
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.encodings = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_encodings.json"), "r"))


        self.model.to(self.device)  # send model to GPU

    def batchToGPU(self, batch: dict) -> dict:
        """
        Sends batch tuple to GPU.

        Args:
            batch (dict): batch to be sent to GPU.

        Returns:
            dict: same batch that was passed as argument but values are in GPU.
        """
        processed_batch = {"return_loss": True}
        print("original images used:", batch["img_id"][:3])
        processed_batch["input_ids"] = batch["input_ids"].to(self.device)#, non_blocking=True) 
        processed_batch["attention_mask"] = batch["attention_mask"].to(self.device)#, non_blocking=True) 
        processed_batch["pixel_values"] = batch["pixel_values"].to(self.device)#, non_blocking=True)
        return processed_batch
    
    def getBestModel(self, validation_losses: dict) -> int:
        if validation_losses != {}:
            return max(validation_losses.values())
        else:
            return -1

    def saveModel(self, current_epoch: int, validation_losses: dict) -> None:
        """
        Updates the currently saved models to only keep the current best model
        Args:
            epoch (int): corresponding epoch of the model that it is being saved.
            epochs (int): data related to all the previous epochs.

        Returns:
            str: path of the saved model file.
        """
        if self.getBestModel(validation_losses) == current_epoch:
            for model in os.listdir(self.run_folder):
                if model.endswith(".pt"):
                    # delete the previous best model
                    os.remove(os.path.join(self.run_folder, model))
            # save the current model
            file_path = self.run_file + "cp-" + current_epoch + ".pt"
            wandb.run.summary["best model"] = file_path
            self.model.save_pretrained(file_path)
    
    def trainPatience(self, validation_losses):
        highest_validation_epoch = self.getBestModel(validation_losses)
        if highest_validation_epoch == -1 or self.patience == 0:
            return True
        else:
            return len(validation_losses) <= highest_validation_epoch + self.patience

    def train(self) -> None:
        epoch_count = 1
        validation_losses = {}
        
        while epoch_count <= self.limit_epochs and self.trainPatience(validation_losses):
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- started epoch", epoch_count)     
            progress_bar = tqdm(range(len(self.train_loader)))
            # train loop
            running_loss = 0.0
            step = 0
            for batch in self.train_loader:
                # train step
                batch = self.batchToGPU(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model(**batch)["loss"]
                running_loss += loss.item()
                total_loss += running_loss
                loss.backward()
                self.optimizer.step()
                # /train step
                if step % self.logging_steps == 0 and step > 0:
                    wandb.log({"train/loss": running_loss/self.logging_steps}, commit=False)
                    running_loss = 0.0
                step += 1
                progress_bar.update(1)


            progress_bar.close()

            # validate
            validation_loss = self.validate()
            validation_losses[epoch_count] = validation_loss
            self.lr_scheduler.step(validation_loss) # update learning rate
            wandb.log({"validation/loss": validation_loss, "epochs": epoch_count})

            # finished epoch
            epoch_count += 1
            self.saveModel(epoch_count, validation_losses)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- finished epoch", epoch_count)     

    def validate(self) -> Tuple[dict, float]:
        """
        Evaluates the trainer's model performance (accuracy) with the given validation dataset.

        Returns:
            Tuple[dict, float]: Tuple with a dictionary with the performance metrics (accuracy) for the trainer's model with the validation dataset
            and the validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        progress_bar = tqdm(range(len(self.test_loader)))
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.batchToGPU(batch)
                running_loss += self.model(**batch)["loss"].item()
                progress_bar.update(1)
        progress_bar.close()
        self.model.train()
        return running_loss.item()/len(self.validation_loader)

    def test(self) -> Union[dict, tuple]:
        """
        Evaluates the trainer's model performance (accuracy) with the test dataset.

        If the dataset being used is the RSVQA-HR also evaluates the trainer's model performance with the Philadelphia teste dataset.

        Returns:
            Union[dict, tuple] : If the dataset used is the RSVQA-HR returns a tuple of two dictionaries with the performance metrics for each test dataset.
            If not, only returns one dictionary with the performance metrics (accuracy) for the trainer's model with the test dataset.

        """
        print("Computing metrics for test dataset")
        total_loss = 0.0
        running_loss = 0.0
        step = 0
        self.model.eval()
        progress_bar = tqdm(range(len(self.test_loader)))
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.batchToGPU(batch)
                loss = self.model(**batch)["loss"].item()
                running_loss += loss
                total_loss += loss
                if step % self.logging_steps == 0 and step > 0:
                    wandb.log({"test/loss": running_loss/self.logging_steps})
                    running_loss = 0.0
                step += 1
                progress_bar.update(1)
        progress_bar.close()
        wandb.run.summary["average test loss"] = total_loss/len(self.dataset_loader)
   
    def run(self):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- started run")     
        self.train()
        self.test()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- finished run")     
    
    
    
    def testing(self):
        batch = next(iter(self.train_loader))
        batch = self.batchToGPU(batch)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_0.jpg"), "w") as image:
            save_image(batch["pixel_values"][0], image)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_1.jpg"), "w") as image:
            save_image(batch["pixel_values"][1], image)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_2.jpg"), "w") as image:
            save_image(batch["pixel_values"][2], image)
        print("loss value:", self.model(**batch)["loss"].item()) 
        print("end:")
        print("\ttorch.cuda.memory_allocated: %fMiB"%(torch.cuda.memory_allocated(0)/1024/1024))
        print("\ttorch.cuda.memory_reserved: %fMiB"%(torch.cuda.memory_reserved(0)/1024/1024))
        file_path = self.run_folder + "/cp-1"
        print(file_path)
        self.model.save_pretrained(file_path)

        exit()


