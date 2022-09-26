import json
import os
from datetime import datetime
from typing import Tuple, Union

import datasets
import torch
import wandb
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor

from Model import CLIPxRSVQA


class Trainer:
    def __init__(self, limit_epochs: int = 25, batch_size: int = 120, patience: int = 3, dataset_name: str = None,
                 device: torch.device = torch.device("cpu"), load_model=False, model_path=None) -> None:
        self.run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.limit_epochs = limit_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dataset_name = dataset_name
        self.device = device

        self.images_path = os.path.join("datasets", dataset_name, "images")
        self.dataset = datasets.load_from_disk(os.path.join("datasets", dataset_name, "dataset"))
        self.encodeDatasetLabels()
        self.train_loader = torch.utils.data.DataLoader(self.dataset["train"], batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)

        self.test_loader = torch.utils.data.DataLoader(self.dataset["test"], batch_size=self.batch_size,
                                                       shuffle=False, num_workers=2)
        if dataset_name == "RSVQA-HR":
            self.test_phili_loader = torch.utils.data.DataLoader(self.dataset["test_phili"], batch_size=self.batch_size,
                                                                 shuffle=False, num_workers=2)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset["validation"], batch_size=self.batch_size,
                                                             shuffle=False, num_workers=2)

        self.input_processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
        self.model = CLIPxRSVQA(num_labels=len(self.label2id))

        if load_model:
            loaded_data = torch.load(model_path)
            self.model.load_state_dict(loaded_data["model_state_dict"])
            self.id2label = loaded_data["id2label"]
            self.label2id = loaded_data["label2id"]
            print("A model has been loaded from file", model_path)

        self.lr = 1e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_patience = 10
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=self.lr_patience)

        config = {
            "initial_learning_rate": self.lr,
            "limit epochs": self.limit_epochs,
            "batch_size": self.batch_size,
            "dataset": self.dataset_name,
            "train patience": self.patience,
            "learning rate patience": self.lr_patience,
        }
        if load_model:
            wandb.init(project="CLIPxRSVQA", job_type="eval",
                       name=self.run_name, tags=[dataset_name], config=config)
        else:
            wandb.init(project="CLIPxRSVQA", job_type="train",
                       name=self.run_name, tags=[dataset_name], config=config)
        self.model.freeze_vision()
        self.model.to(self.device)  # send model to GPU

    def encodeDatasetLabels(self) -> None:
        """
        Create translation dictionaries for the labels from the dataset.

        Translates the labels from text to an id - from 1 to N, where N is the number of possible labels in the dataset.
        """
        self.labels = list(set(self.dataset["train"]["answer"]))
        self.label2id = {}
        self.id2label = {}
        count = 0
        for label in self.labels:
            self.label2id[label] = count
            self.id2label[count] = label
            count += 1

    def prepareBatch(self, batch: dict) -> dict:
        """
        Prepares batch for model training. Sends batch to GPU. Returns the processed batch.

        Args:
            batch (dict): Batch given by torch.utils.data.DataLoader.

        Returns:
            dict: processed batch in GPU, ready to be fed to model.
        """
        # create training batch
        img_paths = []
        # RSVQAxBEN image folder is distributed across subfolders
        if self.dataset_name == "RSVQAxBEN":
            batch["img_id"] = [os.path.join(str(img_id // 2000), str(img_id)) for img_id in batch["img_id"]]

        for img_id in batch["img_id"].tolist():
            if os.path.exists(os.path.join(self.images_path, str(img_id) + ".jpg")):
                img_paths.append(os.path.join(self.images_path, str(img_id) + ".jpg"))

        imgs_to_encode = [Image.open(img) for img in img_paths]

        # process the entire batch at once with padding for dynamic padding
        processed_batch = self.input_processor(
            text=batch["question"], images=imgs_to_encode, padding=True, return_tensors="pt")
        # processed_batch = {**dict(self.tokenizer(batch["question"], return_tensors="pt", padding=true))}
        # processed_batch = {**processed_batch, **dict(self.feature_extractor(imgs_to_encode, return_tensors="pt"))}
        del imgs_to_encode  # free up memory from imgs
        processed_input = {**{"labels": torch.tensor([self.label2id[label]
                                                      for label in batch["answer"]])}, **dict(processed_batch)}

        # send tensors to GPU
        for key in processed_input:
            processed_input[key] = processed_input[key].to(self.device)
        return processed_input

    def test(self) -> Union[dict, tuple]:
        """
        Evaluates the trainer's model performance (accuracy) with the test dataset.

        If the dataset being used is the RSVQA-HR also evaluates the trainer's model performance with the Philadelphia teste dataset.

        Returns:
            Union[dict, tuple] : If the dataset used is the RSVQA-HR returns a tuple of two dictionaries with the performance metrics for each test dataset.
            If not, only returns one dictionary with the performance metrics (accuracy) for the trainer's model with the test dataset.

        """
        print("Computing metrics for test dataset")

        progress_bar = tqdm(range(len(self.test_loader)))
        metrics = {"overall": datasets.load_metric("accuracy")}
        for question_type in list(set(self.dataset["test"]["category"])):
            metrics[str(question_type)] = datasets.load_metric("accuracy")
        self.model.eval()
        for batch in self.test_loader:
            processed_input = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**processed_input)
            logits = output["logits"]
            predictions = torch.argmax(logits, dim=-1)
            for (prediction, category, ground_truth) in zip(predictions, batch["category"], processed_input["labels"]):
                metrics[category].add(prediction=prediction, reference=ground_truth)
                metrics["overall"].add(prediction=prediction, references=ground_truth)
            progress_bar.update(1)
        total_accuracy = 0
        for category in metrics:
            metrics[category] = metrics[category].compute()
            wandb.run.summary["test - " + category + " accuracy"] = metrics[category]["accuracy"]
            if category != "overall":
                total_accuracy += metrics[category]["accuracy"]
        metrics["average"] = total_accuracy / (len(metrics) - 1)
        wandb.run.summary["test - average accuracy"] = metrics["average"]
        progress_bar.close()

        if self.dataset_name == "RSVQA-HR":
            print("Computing metrics for test Philadelphia dataset")

            progress_bar = tqdm(range(len(self.test_phili_loader)))
            metrics_phili = {"overall": datasets.load_metric("accuracy")}
            for question_type in list(set(self.dataset["test"]["category"])):
                metrics_phili[str(question_type)] = datasets.load_metric("accuracy")
            for batch in self.test_phili_loader:
                processed_input = self.prepareBatch(batch)
                with torch.no_grad():
                    output = self.model(**batch)
                logits = output["logits"]
                predictions = torch.argmax(logits, dim=-1)
                for (prediction, category, ground_truth) in zip(predictions, batch["category"], processed_input["labels"]):
                    metrics_phili[category].add(prediction=prediction, reference=ground_truth)
                    metrics_phili["overall"].add(prediction=prediction, references=ground_truth)
                progress_bar.update(1)

            total_accuracy = 0
            for category in metrics_phili:
                metrics_phili[category] = metrics_phili[category].compute()
                wandb.run.summary["test phili - " + category + " accuracy"] = metrics_phili[category]["accuracy"]
                if category != "overall":
                    total_accuracy += metrics_phili[category]["accuracy"]
            metrics_phili["average"] = total_accuracy / (len(metrics_phili) - 1)
            wandb.run.summary["test phili - average accuracy"] = metrics_phili["average"]
            progress_bar.close()

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
        for batch in self.validation_loader:
            processed_input = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**processed_input)
            logits = output["logits"]
            predictions = torch.argmax(logits, dim=-1)
            metrics.add_batch(predictions=predictions, references=processed_input["labels"])
        self.model.train()
        return metrics.compute(), output["loss"].item()

    def trainPatience(self, epochs: dict) -> bool:
        """
        Checks if the best model during training was achieved or not. If the patience is 0, patience check is ignored. 

        Args:
            epochs (int): Number of epochs that already happened during training.

        Returns:
            bool: Returns True if the limit for the training hasn't been reached yet.
        """
        highest_validation_epoch = self.getBestModel(epochs)
        if highest_validation_epoch == -1 or self.patience == 0:
            return True
        else:
            return len(epochs) < highest_validation_epoch + self.patience

    def getBestModel(self, epochs: dict) -> int:
        """
        Given a dictionary with the epochs data returns the epoch with the best model (highest validation metrics).

        Args:
            epochs (dict): Dictionary with the metrics for each training epoch..

        Returns:
            int: Epoch with the highest validation accuracy in the given epochs.
        """
        if epochs != {}:
            return max(epochs, key=lambda epoch: epochs[epoch]["validation metrics"]["accuracy"])
        else:
            return -1

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

        if self.dataset_name == "RSVQA-HR":
            log_to_write["test metrics"], log_to_write["test phili metrics"] = self.test()
        else:
            log_to_write["test metrics"] = self.test()

        log_to_write["best model"] = os.path.join(
            self.log_folder_path, "epoch_" + str(self.getBestModel(epochs)) + "_model.pth")
        wandb.run.summary["best model"] = log_to_write["best model"]
        log_to_write["epochs"] = epochs

        with open(os.path.join(self.log_folder_path, self.run_name.replace(" ", "_").replace(":", "_").replace("-", "_") + ".json"), "w") as log_file:
            json.dump(log_to_write, log_file, indent=4)

        return log_to_write

    def saveModel(self, epoch: int, epochs: int) -> None:
        """
        Updates the currently saved models to only keep the current best model
        Args:
            epoch (int): corresponding epoch of the model that it is being saved.
            epochs (int): data related to all the previous epochs.

        Returns:
            str: path of the saved model file.
        """

        if self.getBestModel(epochs) == epoch:
            for model in os.listdir(self.log_folder_path):
                if model.endswith(".pt"):
                    # delete the previous best model
                    os.remove(os.path.join(self.log_folder_path, model))
            file_path = os.path.join(self.log_folder_path, "epoch_" + str(epoch) + "_model.pt")
            torch.save({"label2id": self.label2id, "id2label": self.id2label,
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

        epochs = {}
        epoch_count = 1

        # training loop
        while epoch_count <= self.limit_epochs and self.trainPatience(epochs):
            epoch_start_time = datetime.now()
            print(epoch_start_time.strftime("%Y-%m-%d %H:%M:%S"), "- Started epoch", epoch_count)
            epoch_progress = tqdm(range(len(self.train_loader)))
            running_loss = 0.0
            train_metrics = {"overall": datasets.load_metric("accuracy")}
            for question_type in list(set(self.dataset["train"]["category"])):
                train_metrics[str(question_type)] = datasets.load_metric("accuracy")
            epochs[epoch_count] = {}

            # train
            for batch in self.train_loader:
                # encode batch and feed it to model
                processed_input = self.prepareBatch(batch)
                self.optimizer.zero_grad()

                output = self.model(**processed_input)
                output["loss"].backward()
                running_loss += output["loss"].item()
                self.optimizer.step()

                logits = output["logits"]

                predictions = torch.argmax(logits, dim=-1)
                for (prediction, category, ground_truth) in zip(predictions, batch["category"], processed_input["labels"]):
                    train_metrics[category].add(prediction=prediction, reference=ground_truth)
                    train_metrics["overall"].add(prediction=prediction, references=ground_truth)
                epoch_progress.update(1)

            # current training loss and accuracy for the epoch
            to_log = {"epochs": epoch_count}
            epoch_finish_time = datetime.now()
            to_log["learning rate"] = epochs[epoch_count]["learning rate"] = self.lr_scheduler.optimizer.param_groups[0]["lr"]

            for category in train_metrics:
                train_metrics[category] = train_metrics[category].compute()
                to_log["train - " + category + " accuracy"] = train_metrics[category]["accuracy"]
            epochs[epoch_count]["train metrics"] = train_metrics

            to_log["train - loss"] = epochs[epoch_count]["train loss"] = running_loss/len(self.train_loader)

            # validate the training epoch
            epochs[epoch_count]["validation metrics"], epochs[epoch_count]["validation loss"] = self.validate()
            to_log["validation - overall accuracy"] = epochs[epoch_count]["validation metrics"]["accuracy"]
            to_log["validation - loss"] = epochs[epoch_count]["validation loss"]

            epochs[epoch_count]["elapsed time"] = str((epoch_finish_time - epoch_start_time)).split(".")[0]

            # save the model state if this epoch has the current best model
            self.saveModel(epoch_count, epochs)

            # update learning rate
            self.lr_scheduler.step(epochs[epoch_count]["validation loss"])
            wandb.log(to_log)

            epoch_progress.close()
            print(epoch_finish_time.strftime("%Y-%m-%d %H:%M:%S"), "- Finished epoch", epoch_count)
            epoch_count += 1

        self.saveTrain(epochs)
