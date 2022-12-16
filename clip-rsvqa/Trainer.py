import json
import os
from datetime import datetime
from typing import Tuple, Union
import Models
import torch
import wandb
import H5Datasets
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss


class Trainer:
    def __init__(self, batch_size: int, limit_epochs: int, patience: int, lr_patience: int, freeze: bool, dataset_name: str, model_type: str,
                 device: torch.device, load_model: bool = False, model_path: str = None) -> None:
        self.run_config = {
            "batch size": batch_size,
            "limit epochs": limit_epochs,
            "patience": patience,
            "intial learning rate": 1e-4,
            "learning rate patience": lr_patience,
            "dataset": dataset_name,
            "logging steps": 100,
            "freeze CLIP Vision": freeze,
            "model architecure": model_type,
            "model aspect ratio": (1, 32),
            "optimizer": "AdamW"
        }
        self.run_name = f"{dataset_name:s}:bs{self.run_config['batch size']:d}-lr{self.run_config['initial learning rate']:.0e}-\
                        lrp{self.run_config['learning rate patience']:d}-adamw"
        if load_model:
            self.job_type = "eval"
            wandb.init(project="CLIPxRSVQA", job_type="eval",
                       name=self.run_name, config=self.run_config)
        else:
            self.job_type = "train"
            wandb.init(project="CLIPxRSVQA", job_type="train",
                       name=self.run_name, config=self.run_config)
        self.run_name = f"{wandb.run.id:s}-{self.run_name:s}"
        self.run_folder = os.path.join(
            "saved-models", self.run_config["dataset"], self.run_name)
        os.makedirs(self.run_folder)
        self.batch_size = self.run_config["batch size"]
        self.limit_epochs = self.run_config["limit epochs"]
        self.patience = self.run_config["patience"]
        self.dataset_name = self.run_config["dataset"]
        self.lr = self.run_config["initial learning rate"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr)
        self.lr_patience = self.run_config["learning rate patience"]
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience)
        self.loss_fcn = CrossEntropyLoss()
        self.freeze = self.run_config["freeze CLIP Vision"]
        self.logging_steps = self.run_config["logging steps"]
        self.device = device
        if self.freeze:
            self.model.freeze_vision()

        if self.dataset_name == "RSVQA-LR":
            self.train_dataset = H5Datasets.RsvqaDataset("RSVQA-LR", "train", self.run_config["model architecture"])
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=6, pin_memory=True)
            self.validation_dataset = H5Datasets.RsvqaDataset("RSVQA-LR", "validation", self.run_config["model architecture"])
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=6, pin_memory=True)
            self.test_dataset = H5Datasets.RsvqaDataset("RSVQA-LR", "test", self.run_config["model architecture"])
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=6, pin_memory=True)
            metadata = json.load(open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_metadata.json"), "r"))
            self.id2label = metadata["id2label"]
            self.label2id = metadata["label2id"]

        elif self.dataset_name == "RSVQA-HR":
            self.train_dataset = H5Datasets.RsvqaDataset("RSVQA-HR", "train", self.run_config["model architecture"])
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=6, pin_memory=True)
            self.validation_dataset = H5Datasets.RsvqaDataset("RSVQA-HR", "validation", self.run_config["model architecture"])
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=6, pin_memory=True)
            self.test_dataset = H5Datasets.RsvqaDataset("RSVQA-HR", "test", self.run_config["model architecture"])
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=6, pin_memory=True)
            self.test_phili_dataset = H5Datasets.RsvqaDataset("RSVQA-HR", "test_phili", self.run_config["model architecture"])
            self.test_phili_loader = torch.utils.data.DataLoader(self.test_phili_dataset, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=6, pin_memory=True)
            metadata = json.load(open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_metadata.json"), "r"))
            self.id2label = metadata["id2label"]
            self.label2id = metadata["label2id"]

        elif self.dataset_name == "RSVQAxBEN":
            self.train_dataset = H5Datasets.RsvqaDataset("RSVQAxBEN", "train", self.run_config["model architecture"])
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, num_workers=6, pin_memory=True)
            self.validation_dataset = H5Datasets.RsvqaDataset("RSVQAxBEN", "validation", self.run_config["model architecture"])
            self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=6, pin_memory=True)
            self.test_dataset = H5Datasets.RsvqaDataset("RSVQAxBEN", "test", self.run_config["model architecture"])
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           shuffle=False, num_workers=6, pin_memory=True)
            metadata = json.load(open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_metadata.json"), "r"))
            metadata = json.load(open(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben_metadata.json"), "r"))
            self.id2label = metadata["id2label"]
            self.label2id = metadata["label2id"]        

        if self.run_config["model architecture"] == "baseline":
            # model aspect ratio = (n_layers, n_heads)
            self.model = Models.Baseline(
                num_labels=self.train_dataset.num_labels["total"], model_aspect_ratio=self.run_config["model aspect ratio"])
        elif self.run_config["model architecture"] == "patching":
            # model aspect ratio = (n_layers, n_heads)
            self.model = Models.Patching(
                num_labels=self.train_dataset.num_labels["total"], model_aspect_ratio=self.run_config["model aspect ratio"])
        self.model.name = self.run_config["model architecture"]

        if load_model:
            self.load_model(model_path)

        self.model.to(self.device)  # send model to GPU

    def load_model(self, model_path: str) -> None:
        """
        Loads the model weights and label encodings from .pt file

        Args:
            model_path (str): Path to the model.pt file
        """
        loaded_data = torch.load(model_path)
        self.model.load_state_dict(
            loaded_data["model_state_dict"])
        self.id2label = loaded_data["id2label"]
        self.label2id = loaded_data["label2id"]
        print("A model has been loaded from file", model_path)

    def trainPatience(self, validation_metrics: dict) -> bool:
        """
        Checks if the best model during training was achieved or not. If the patience is 0, patience check is ignored.

        Args:
            epochs (int): Number of epochs that already happened during training.

        Returns:
            bool: Returns True if the limit for the training hasn't been reached yet.
        """
        highest_validation_epoch = self.getBestModel(
            validation_metrics)
        if highest_validation_epoch == -1 or self.patience == 0:
            return True
        else:
            return len(validation_metrics) <= highest_validation_epoch + self.patience

    def getBestModel(self, validation_metrics: dict[int, dict[str, float]]) -> int:
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
        batch["input_ids"] = batch["input_ids"].to(
            self.device, non_blocking=True)
        batch["attention_mask"] = batch["attention_mask"].to(
            self.device, non_blocking=True)
        batch["pixel_values"] = batch["pixel_values"].to(
            self.device, non_blocking=True)
        batch["label"] = batch["label"]
        batch["category"] = batch["category"]
        return batch

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
            for model in os.listdir(self.run_folder):
                if model.endswith(".pt"):
                    # delete the previous best model
                    os.remove(os.path.join(
                        self.run_folder, model))
            self.model_path = os.path.join(
                self.run_folder, f"cp-{current_epoch:d}")
            wandb.run.summary["best model"] = self.model_path
            wandb.run.summary["best model epoch"] = current_epoch
            torch.save({"label2id": self.label2id, "id2label": self.id2label,
                       "model_state_dict": self.model.state_dict()}, self.model_path)

    def test(self) -> Union[dict, tuple]:
        """
        Evaluates the trainer's model performance (accuracy) with the test dataset.

        If the dataset being used is the RSVQA-HR also evaluates the trainer's model performance with the Philadelphia test dataset.

        Returns:
            Union[dict, tuple] : If the dataset used is the RSVQA-HR returns a tuple of two dictionaries with the performance metrics for each test dataset.
            If not, only returns one dictionary with the performance metrics (accuracy) for the trainer's model with the test dataset.

        """
        self.load_model(self.model_path)
        print("Computing metrics for test dataset")
        self.model.eval()

        metrics = {"overall": {"accuracy": Accuracy(
            task="multiclass", num_classes=self.test_dataset.num_possible_answers["total"])}}
        for question_type in self.test_dataset.categories:
            metrics[question_type] = {"accuracy": Accuracy(
                task="multiclass", num_classes=self.test_dataset.num_possible_answers[question_type])}

        progress_bar = tqdm(range(len(self.test_loader)))
        for batch in self.test_loader:
            batch = self.batchToGPU(batch)
            with torch.no_grad():
                logits = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
            predictions = torch.argmax(logits, dim=-1)
            for (prediction, question_type, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                metrics[question_type]["accuracy"](
                    prediction, ground_truth)
                metrics["overall"]["accuracy"](
                    prediction, ground_truth)
            progress_bar.update(1)

        total_accuracy = 0
        for question_type in metrics:
            wandb.run.summary["test - " + question_type +
                              " accuracy"] = metrics[question_type]["accuracy"].compute()
            if question_type != "overall":
                total_accuracy += wandb.run.summary["test - " +
                                                    question_type + " accuracy"]
        wandb.run.summary["test - average accuracy"] = total_accuracy / \
                                                        (len(metrics) - 1)
        progress_bar.close()

        if self.dataset_name == "RSVQA-HR":
            print(
                "Computing metrics for test Philadelphia dataset")
            metrics_phili = {"overall": {"accuracy": Accuracy(
                task="multiclass", num_classes=self.test_dataset.num_possible_answers["total"])}}
            for question_type in self.test_dataset.categories:
                metrics_phili[question_type] = {"accuracy": Accuracy(
                    task="multiclass", num_classes=self.test_dataset.num_possible_answers[question_type])}

            progress_bar = tqdm(
                range(len(self.test_phili_loader)))
            for batch in self.test_phili_loader:
                batch = self.batchToGPU(batch)
                with torch.no_grad():
                    logits = self.model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                predictions = torch.argmax(logits, dim=-1)
                for (prediction, question_type, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                    metrics_phili[question_type]["accuracy"](
                        prediction, ground_truth)
                    metrics_phili["overall"]["accuracy"](
                        prediction, ground_truth)
                progress_bar.update(1)

            total_accuracy = 0
            for question_type in metrics_phili:
                wandb.run.summary["test phili - " + question_type +
                                  " accuracy"] = metrics_phili[question_type]["accuracy"].compute()
                if question_type != "overall":
                    total_accuracy += wandb.run.summary["test phili - " +
                                                        question_type + " accuracy"]
            wandb.run.summary["test phili - average accuracy"] = total_accuracy / \
                (len(metrics_phili) - 1)
            progress_bar.close()

    def validate(self) -> Tuple[dict, float]:
        """
        Evaluates the trainer's model performance (accuracy) with the given validation dataset.

        Returns:
            Tuple[dict, float]: Tuple with a dictionary with the performance metrics (accuracy) for the trainer's model with the validation dataset
            and the validation loss.
        """
        self.model.eval()
        metric = Accuracy(task="multiclass",
                          num_classes=self.validation_dataset.num_possible_answers["total"])
        running_loss = 0.0
        with torch.no_grad():
            for batch in self.validation_loader:
                batch = self.batchToGPU(batch)
                logits = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                running_loss += self.loss_fcn(
                    logits, batch["label"])
                predictions = torch.argmax(logits, dim=-1)
                metric(predictions, batch["label"])
        self.model.train()
        return metric.compute(), running_loss.item()/len(self.validation_loader)

    def train(self) -> None:
        """
        Training loop.
        The training loop has two different stop conditions:
            1) a limit of epochs is reached;
            2) if the interval of epochs between the highest validation accuracy and current epoch is higher than a threshold (default is 3).
            If the threshold value is 0 this stop condition is ignored.
        Once the training loop is complete a log is saved.
        """

        epoch_count = 1
        validation_metrics = {}
        # training loop
        while epoch_count <= self.limit_epochs and self.trainPatience(validation_metrics):
            print(datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"), "- Started epoch", epoch_count)
            train_metrics = {"overall": {"accuracy": Accuracy(
                task="multiclass", num_classes=self.train_dataset.num_possible_answers["total"])}}
            for question_type in self.train_dataset.categories:
                train_metrics[question_type] = {"accuracy": Accuracy(
                    task="multiclass", num_classes=self.train_dataset.num_possible_answers[question_type])}
            validation_metrics[epoch_count] = {}

            epoch_progress = tqdm(
                range(len(self.train_loader)))
            # per epoch loop
            running_loss = 0.0
            step = 0
            for batch in self.train_loader:
                # train step
                batch = self.batchToGPU(batch)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                loss = self.loss_fcn(logits, batch["label"])
                loss.backward()
                running_loss += loss
                # gradient acummulation every 2 batches to match the same batch size for both model architecures
                if self.run_config["model architecure"] == "patching" and step % 2 == 0 and step > 0: # TODO needs to fixed lol
                    self.optimizer.step()
                elif self.run_config["model architecure"] == "baseline":
                    self.optimizer.step()

                predictions = torch.argmax(logits, dim=-1)
                for (prediction, question_type, ground_truth) in zip(predictions, batch["category"], batch["label"]):
                    train_metrics[question_type]["accuracy"](
                        prediction, ground_truth)
                    train_metrics["overall"]["accuracy"](
                        prediction, ground_truth)
                # /train step
                if step % self.logging_steps == 0 and step > 0:
                    wandb.log(
                        {"train/loss": running_loss/self.logging_steps})
                    running_loss = 0.0
                epoch_progress.update(1)
                step += 1

            # log train results
            wandb.log(
                {"learning rate": self.lr_scheduler.optimizer.param_groups[0]["lr"]}, commit=False)

            total_train_accuracy = 0
            for question_type in train_metrics:
                accuracy = train_metrics[question_type]["accuracy"].compute(
                )
                total_train_accuracy += accuracy
                wandb.log({"train/" + question_type +
                          " accuracy": accuracy}, commit=False)
            wandb.log(
                {"train/overall accuracy": train_metrics["overall"]["accuracy"].compute()}, commit=False)
            wandb.log({"train/average accuracy": total_train_accuracy /
                      len(train_metrics)-1}, commit=False)

            # validate
            validation_metrics[epoch_count]["accuracy"], validation_metrics[epoch_count]["loss"] = self.validate(
            )
            wandb.log({"validation/overall accuracy": validation_metrics[epoch_count]["accuracy"],
                      "validation/loss": validation_metrics[epoch_count]["loss"]}, commit=False)
            self.lr_scheduler.step(
                validation_metrics[epoch_count]["loss"])

            # save the model state if this epoch has the current best model
            self.saveModel(epoch_count, validation_metrics)
            # finish epoch
            epoch_progress.close()
            print(datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"), "- Finished epoch", epoch_count)
            wandb.log({"epochs": epoch_count})
            epoch_count += 1
    

    def run(self):
        """
        Main loop. Trains the model and then evaluates it.
        """
        print(datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), "- started run")
        if self.job_type == "train":
            self.train()
            self.test()
        elif self.job_type == "eval":
            self.test()
        print(datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"), "- finished run")
