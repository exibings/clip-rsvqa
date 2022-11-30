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
    def __init__(self):
        self.model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.batch_size = 100
        self.limit_epochs = 100
        self.patience = 20
        self.lr = 5e-5
        self.lr_patience = 10
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer, "min", patience=self.lr_patience)
        
        self.dataset_name = "NWPU-Captions"
        self.train_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "train")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6, pin_memory=True)
        #self.validation_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "validation")
        #self.validation_loader = torch.utils.Data.dataLoader(self.validation_dataset, collate_fn=self.dataCollator, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        #self.test_dataset = H5Datasets.NwpuCaptionsDataset(os.path.join("datasets", "NWPU-Captions", "nwpu_captions.h5"), "test")
        #self.test_loader = torch.utils.data.DataLoader(self.test_dataset, collate_fn=self.dataCollator, batch_size=self.batch_size, shuffle=False, num_workers=6, pin_memory=True)
        self.encodings = json.load(open(os.path.join("datasets", "NWPU-Captions", "nwpu_captions_encodings.json"), "r"))
        
        """
        wandb_config = {
            "initial_learning_rate": self.lr,
            "limit epochs": self.limit_epochs,
            "batch_size": self.batch_size,
            "dataset": self.dataset_name,
            "train patience": self.patience,
            "learning rate patience": self.lr_patience
        }
        """

        #wandb.init(project="CLIPxRSVQA", job_type="pre-train", name=self.run_name, config=wandb_config)
        print(self.run_name, "batch size:", self.batch_size)
        print("start:")
        print("\ttorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        self.model.to(self.device)  # send model to GPU
        print("model sent to GPU:")
        print("\ttorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))



    def batchToGPU(self, batch: dict) -> dict:
        """
        Sends batch tuple to GPU.

        Args:
            batch (dict): batch to be sent to GPU.

        Returns:
            dict: same batch that was passed as argument but values are in GPU.
        """
        processed_batch = {"return_loss": True}
        print("original images used:", batch["img_id"][:2])
        processed_batch["input_ids"] = batch["input_ids"].to(self.device)#, non_blocking=True) 
        processed_batch["attention_mask"] = batch["attention_mask"].to(self.device)#, non_blocking=True) 
        processed_batch["pixel_values"] = batch["pixel_values"].to(self.device)#, non_blocking=True)
        print("batch to gpu:")
        print("\ttorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        return processed_batch
    
    def train(self) -> None:
        #TODO
        progress_bar = tqdm(range(len(self.train_loader)))
        for batch in self.train_loader:
            progress_bar.update(1)
        return 

    def validate(self) -> Tuple[dict, float]:
        """
        Evaluates the trainer's model performance (accuracy) with the given validation dataset.

        Returns:
            Tuple[dict, float]: Tuple with a dictionary with the performance metrics (accuracy) for the trainer's model with the validation dataset
            and the validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.batchToGPU(batch)
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                running_loss += self.loss_fcn(logits, batch["label"]) # TODO
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


        self.model.eval()
        progress_bar = tqdm(range(len(self.test_loader)))
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.batchToGPU(batch)
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                running_loss += self.loss_fcn(logits, batch["label"]) # TODO
            progress_bar.update(1)
        return running_loss.item()/len(self.test_loader)
    
    '''
    def train(self) -> None:
        """
        Training loop.
        The training loop has two different stop conditions:
            1) a limit of epochs is reached;
            2) if the interval of epochs between the highest validation accuracy and current epoch is higher than a threshold (default is 3).
            If the threshold value is 0 this stop condition is ignored.
        Once the training loop is complete a log is saved.
        """

        # create log folder for the training session
        self.log_folder_path = os.path.join("logs", self.dataset_name, self.run_name.replace(
            " ", "_").replace(":", "_").replace("-", "_"))
        os.makedirs(self.log_folder_path)

        validation_metrics = {}
        epoch_count = 1

        # training loop
        while epoch_count <= self.limit_epochs and self.trainPatience(validation_metrics):
            epoch_start_time = datetime.now()
            print(epoch_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                  "- Started epoch", epoch_count)
            running_loss = 0.0
            train_metrics = {"overall": datasets.load_metric("accuracy")}
            for question_type in self.train_dataset.categories:
                train_metrics[question_type] = datasets.load_metric("accuracy")
            validation_metrics[epoch_count] = {}

            epoch_progress = tqdm(range(len(self.train_loader)))
            # train
            for batch in self.train_loader:
                batch = self.batchToGPU(batch)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                loss = self.loss_fcn(logits, batch["label"])
                loss.backward()
                running_loss += loss
                self.optimizer.step()
                predictions = torch.argmax(logits, dim=-1)
                for (prediction, category, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                    train_metrics[category].add(
                        prediction=prediction, reference=ground_truth)
                    train_metrics["overall"].add(
                        prediction=prediction, references=ground_truth)
                epoch_progress.update(1)

            # current training loss and accuracy for the epoch
            to_log = {"epochs": epoch_count}
            to_log["learning rate"] = self.lr_scheduler.optimizer.param_groups[0]["lr"]
            for category in train_metrics:
                to_log["train - " + category +
                       " accuracy"] = train_metrics[category].compute()["accuracy"]
            to_log["train - loss"] = running_loss.item()/len(self.train_loader)

            # validate the training epoch
            validation_metrics[epoch_count]["accuracy"], validation_metrics[epoch_count]["loss"] = self.validate(
            )
            to_log["validation - overall accuracy"] = validation_metrics[epoch_count]["accuracy"]
            to_log["validation - loss"] = validation_metrics[epoch_count]["loss"]

            # save the model state if this epoch has the current best model
            self.saveModel(epoch_count, validation_metrics)
            # update learning rate
            self.lr_scheduler.step(validation_metrics[epoch_count]["loss"])
            wandb.log(to_log)
            epoch_finish_time = datetime.now()
            epoch_progress.close()
            print(epoch_finish_time.strftime("%Y-%m-%d %H:%M:%S"),
                  "- Finished epoch", epoch_count)
            epoch_count += 1
        self.test()

    '''
    def testing(self):
        batch = next(iter(self.train_loader))
        batch = self.batchToGPU(batch)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_0.jpg"), "w") as image:
            save_image(batch["pixel_values"][0], image)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_1.jpg"), "w") as image:
            save_image(batch["pixel_values"][1], image)
        with open(os.path.join("datasets", "NWPU-Captions", "processed_image_2.jpg"), "w") as image:
            save_image(batch["pixel_values"][2], image)
        output = self.model(**batch)
        print("end:")
        print("\ttorch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("\ttorch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("loss:", output["loss"]) 
        exit()



pre_trainer = PreTrainer()
pre_trainer.testing()