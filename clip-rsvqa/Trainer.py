import json
import os
from datetime import datetime
from typing import Tuple, Union
import Models
import datasets
import torch
import wandb
import H5Datasets
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss


class Trainer:
    def __init__(self, limit_epochs: int = 100, batch_size: int = 48, patience: int = 20, lr_patience: int = 10, freeze: bool = True, dataset_name: str = None,
                 device: torch.device = torch.device("cpu"), load_model: bool = False, model_path: str = None, model: str = None) -> None:
        self.run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.limit_epochs = limit_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dataset_name = dataset_name
        self.device = device

        if self.dataset_name == "RSVQA-LR":
            self.train_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "train", model)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=6, pin_memory=True)
            self.validation_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "validation", model)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                shuffle=False, num_workers=6, pin_memory=True)
            self.test_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "test", model)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=6, pin_memory=True)
            self.encodings = json.load(open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_encodings.json"), "r"))

        elif self.dataset_name == "RSVQA-HR":
            self.train_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr.h5"), "train", model)
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=6, pin_memory=True)
            self.validation_dataset = H5Datasets.RsvqaDataset(os.path.join(
                "datasets", "RSVQA-HR", "rsvqa_hr.h5"), "validation", model)
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=6, pin_memory=True)
            self.test_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr.h5"), "test", model)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=6, pin_memory=True)
            self.test_phili_dataset = H5Datasets.RsvqaDataset(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr.h5"), "test_phili", model)
            self.test_phili_loader = torch.utils.data.DataLoader(self.test_phili_dataset, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=6, pin_memory=True)
            self.encodings = json.load(open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_encodings.json"), "r"))
        
        elif self.dataset_name == "RSVQAxBEN":
            #TODO
            pass

        if model == "baseline":
            self.model = Models.Baseline(num_labels=self.train_dataset.num_labels, model_aspect_ratio=(1,32)) # model aspect ratio = (n_layers, n_heads)
        elif model == "patching":
            self.model = Models.Patching(num_labels=self.train_dataset.num_labels, model_aspect_ratio=(1,32)) # model aspect ratio = (n_layers, n_heads)
        self.model.name = model
        if load_model:
            self.load_model(model_path)

        self.lr = 1e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_patience = lr_patience
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience)
        self.loss_fcn = CrossEntropyLoss()
        self.freeze = freeze

        wandb_config = {
            "initial_learning_rate": self.lr,
            "limit epochs": self.limit_epochs,
            "batch_size": self.batch_size,
            "dataset": self.dataset_name,
            "train patience": self.patience,
            "learning rate patience": self.lr_patience,
            "freeze CLIP Vision": self.freeze,
            "model": model,
            "model aspect ratio": self.model.aspect_ratio
        }

        if self.freeze:
            self.model.freeze_vision()

        if load_model:
            wandb.init(project="CLIPxRSVQA", job_type="eval", name=self.run_name, config=wandb_config)
        else:
            wandb.init(project="CLIPxRSVQA", job_type="train", name=self.run_name, config=wandb_config)
        self.model.to(self.device)  # send model to GPU

    def load_model(self, model_path: str) -> None:
        """
        Loads the model weights and label encodings from .pt file

        Args:
            model_path (str): Path to the model.pt file
        """
        loaded_data = torch.load(model_path)
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.id2label = loaded_data["id2label"]
        self.label2id = loaded_data["label2id"]
        print("A model has been loaded from file", model_path)

    def test(self) -> Union[dict, tuple]:
        """
        Evaluates the trainer's model performance (accuracy) with the test dataset.

        If the dataset being used is the RSVQA-HR also evaluates the trainer's model performance with the Philadelphia teste dataset.

        Returns:
            Union[dict, tuple] : If the dataset used is the RSVQA-HR returns a tuple of two dictionaries with the performance metrics for each test dataset.
            If not, only returns one dictionary with the performance metrics (accuracy) for the trainer's model with the test dataset.

        """
        print("Computing metrics for test dataset")

        metrics = {"overall": datasets.load_metric("accuracy")}
        for question_type in self.test_dataset.categories:
            metrics[question_type] = datasets.load_metric("accuracy")
        self.model.eval()

        progress_bar = tqdm(range(len(self.test_loader)))
        for batch in self.test_loader:
            batch = self.batchToGPU(batch)
            with torch.no_grad():
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
            predictions = torch.argmax(logits, dim=-1)
            for (prediction, category, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                metrics[category].add(prediction=prediction, reference=ground_truth)
                metrics["overall"].add(prediction=prediction, references=ground_truth)
            progress_bar.update(1)
        total_accuracy = 0
        progress_bar.close()
        
        for category in metrics:
            wandb.run.summary["test - " + category + " accuracy"] = metrics[category].compute()["accuracy"]
            if category != "overall":
                total_accuracy += wandb.run.summary["test - " + category + " accuracy"]
        wandb.run.summary["test - average accuracy"] = total_accuracy / (len(metrics) - 1)

        if self.dataset_name == "RSVQA-HR":
            print("Computing metrics for test Philadelphia dataset")
            metrics_phili = {"overall": datasets.load_metric("accuracy")}
            for question_type in self.test_phili_dataset.categories:
                metrics_phili[question_type] = datasets.load_metric("accuracy")
            
            progress_bar = tqdm(range(len(self.test_loader)))
            for batch in self.test_phili_loader:
                batch = self.batchToGPU(batch)
                with torch.no_grad():
                    logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                predictions = torch.argmax(logits, dim=-1)
                for (prediction, category, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                    metrics_phili[category].add(prediction=prediction, reference=ground_truth)
                    metrics_phili["overall"].add(prediction=prediction, references=ground_truth)
                progress_bar.update(1)
            total_accuracy = 0
            progress_bar.close()
            
            for category in metrics_phili:
                wandb.run.summary["test phili - " + category + " accuracy"] = metrics_phili[category].compute()["accuracy"]
                if category != "overall":
                    total_accuracy += wandb.run.summary["test phili - " + category + " accuracy"]
            wandb.run.summary["test phili- average accuracy"] = total_accuracy / (len(metrics_phili) - 1)

            return metrics, metrics_phili

        return metrics

    def validate(self) -> Tuple[dict, float]:
        """
        Evaluates the trainer's model performance (accuracy) with the given validation dataset.

        Returns:
            Tuple[dict, float]: Tuple with a dictionary with the performance metrics (accuracy) for the trainer's model with the validation dataset
            and the validation loss.
        """
        metrics = datasets.load_metric("accuracy")
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.batchToGPU(batch)
                logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                running_loss += self.loss_fcn(logits, batch["label"])
                predictions = torch.argmax(logits, dim=-1)
                metrics.add_batch(predictions=predictions, references=batch["label"])
        self.model.train()
        return metrics.compute()["accuracy"], running_loss.item()/len(self.validation_loader)

    def trainPatience(self, validation_metrics: dict) -> bool:
        """
        Checks if the best model during training was achieved or not. If the patience is 0, patience check is ignored.

        Args:
            epochs (int): Number of epochs that already happened during training.

        Returns:
            bool: Returns True if the limit for the training hasn't been reached yet.
        """
        highest_validation_epoch = self.getBestModel(validation_metrics)
        if highest_validation_epoch == -1 or self.patience == 0:
            return True
        else:
            return len(validation_metrics) <= highest_validation_epoch + self.patience

    def getBestModel(self, validation_metrics: dict) -> int:
        """
        Given a dictionary with the epochs data returns the epoch with the best model (highest validation metrics).

        Args:
            epochs (dict): Dictionary with the metrics for each training epoch..

        Returns:
            int: Epoch with the highest validation accuracy in the given epochs.
        """
        if validation_metrics != {}:
            return max(validation_metrics, key=lambda epoch: validation_metrics[epoch]["accuracy"])
        else:
            return -1

    def batchToGPU(self, batch: dict) -> dict:
        """
        Sends batch tuple to GPU.

        Args:
            batch (dict): batch to be sent to GPU.

        Returns:
            dict: same batch that was passed as argument but values are in GPU.
        """
        batch["input_ids"] = batch["input_ids"].to(self.device, non_blocking=True) 
        batch["attention_mask"] = batch["attention_mask"].to(self.device, non_blocking=True) 
        batch["pixel_values"] = batch["pixel_values"].to(self.device, non_blocking=True) 
        batch["label"] = batch["label"].to(self.device, non_blocking=True) 
        return batch

    def writeLog(self, epochs: dict) -> dict:
        """
        Writes a full log of the training session. Includes the following information:
            - run name;
            - dataset;
            - total elapsed time;
            - total number of epochs;
            - initial learning rate;
            - used patience;
            - used batch size;
            - whether resized images were used;
            - test performance metrics;
            - metrics of each training epoch;

        Args:
            epochs (dict): Dictionary with the metrics for each training epoch.

        Returns:
            dict: Dictionary that was writen as a JSON.
        """

        log_to_write = {
            "run name": self.run_name,
            "dataset": self.dataset_name,
            "device": torch.cuda.get_device_name(self.device),
            "total elapsed time": str(datetime.now() - datetime.strptime(self.run_name, "%Y-%m-%d %H:%M:%S")).split(".")[0],
            "total epochs": len(epochs),
            "initial learning rate": self.lr,
            "patience": self.patience,
            "batch size": self.batch_size,
        }
        log_to_write["best model"] = os.path.join(
            self.log_folder_path, "epoch_" + str(self.getBestModel(epochs)) + "_model.pt")

        self.load_model(log_to_write["best model"])
        if self.dataset_name == "RSVQA-HR":
            log_to_write["test metrics"], log_to_write["test phili metrics"] = self.test()
        else:
            log_to_write["test metrics"] = self.test()
        log_to_write["epochs"] = epochs

        with open(os.path.join(self.log_folder_path, self.run_name.replace(" ", "_").replace(":", "_").replace("-", "_") + ".json"), "w") as log_file:
            json.dump(log_to_write, log_file, indent=4)

        return log_to_write

    def saveModel(self, current_epoch: int, validation_metrics: dict) -> None:
        """
        Updates the currently saved models to only keep the current best model
        Args:
            epoch (int): corresponding epoch of the model that it is being saved.
            epochs (int): data related to all the previous epochs.

        Returns:
            str: path of the saved model file.
        """
        if self.getBestModel(validation_metrics) == current_epoch:
            for model in os.listdir(self.log_folder_path):
                if model.endswith(".pt"):
                    # delete the previous best model
                    os.remove(os.path.join(self.log_folder_path, model))
            file_path = os.path.join(self.log_folder_path, "epoch_" + str(current_epoch) + "_model.pt")
            wandb.run.summary["best model"] = file_path
            torch.save({"label2id": self.encodings["label2id"], "id2label": self.encodings["id2label"],
                       "model_state_dict": self.model.state_dict()}, file_path)

    def saveTrain(self, epochs: dict) -> None:
        """
        Saves the complete log of the training, including the best model.

        Args:
            epochs (dict): Dictionary with all the information relevant to the training.
        """
        # save the best model and write the logs
        print("Completed model training in",
              self.writeLog(epochs)["total elapsed time"])

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
            print(epoch_start_time.strftime("%Y-%m-%d %H:%M:%S"), "- Started epoch", epoch_count)
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
                logits = self.model(input_ids=batch["input_ids"], attention_mask = batch["attention_mask"], pixel_values=batch["pixel_values"])
                loss = self.loss_fcn(logits, batch["label"])
                loss.backward()
                running_loss += loss
                self.optimizer.step()
                predictions = torch.argmax(logits, dim=-1)
                for (prediction, category, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                    train_metrics[category].add(prediction=prediction, reference=ground_truth)
                    train_metrics["overall"].add(prediction=prediction, references=ground_truth)
                epoch_progress.update(1)

            # current training loss and accuracy for the epoch
            to_log = {"epochs": epoch_count}
            to_log["learning rate"] = self.lr_scheduler.optimizer.param_groups[0]["lr"]
            for category in train_metrics:
                to_log["train - " + category + " accuracy"] = train_metrics[category].compute()["accuracy"]
            to_log["train - loss"] =  running_loss.item()/len(self.train_loader)

            # validate the training epoch
            validation_metrics[epoch_count]["accuracy"], validation_metrics[epoch_count]["loss"] = self.validate() 
            to_log["validation - overall accuracy"] = validation_metrics[epoch_count]["accuracy"]
            to_log["validation - loss"] = validation_metrics[epoch_count]["loss"]


            # save the model state if this epoch has the current best model
            self.saveModel(epoch_count, validation_metrics)
            # update learning rate
            self.lr_scheduler.step(validation_metrics[epoch_count]["loss"])
            wandb.log(to_log)
            epoch_finish_time = datetime.now()
            epoch_progress.close()
            print(epoch_finish_time.strftime("%Y-%m-%d %H:%M:%S"), "- Finished epoch", epoch_count)
            epoch_count += 1
        self.test()
