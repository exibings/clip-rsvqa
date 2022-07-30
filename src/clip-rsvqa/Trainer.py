import datetime
import json
import os
from copy import deepcopy
from typing import Tuple, Union

import datasets
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from Model import CLIPxRSVQA


class Trainer:
    def __init__(self, limit_epochs: int = 25, batch_size: int = 64, use_resized_images: bool = False, dataset_name: str = None, device: torch.device = torch.device("cpu"), patience: int = 3) -> None:
        self.limit_epochs = limit_epochs
        self.batch_size = batch_size
        self.patience = patience

        self.imagesPath = os.path.join("datasets", dataset_name, "images") if not use_resized_images else os.path.join(
            "datasets", dataset_name, "images", "resized")
        self.dataset_name = dataset_name
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

        self.device = device

        self.input_processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")

        clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")

        self.model = CLIPxRSVQA(config=clip_model.config, num_labels=len(self.label2id), device=self.device)
        self.model.text_model = clip_model.text_model
        self.model.vision_model = clip_model.vision_model
        self.model.visual_projection = clip_model.visual_projection
        self.model.text_projection = clip_model.text_projection
        self.model.logit_scale = clip_model.logit_scale

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # TODO criar verificacoes para nao abusar das GPUs todas
        self.model.to(self.device)  # send model to GPU

        print("Trainer is ready.")

    def encodeDatasetLabels(self) -> None:
        """
        Create translation dictionaries for the labels from the dataset.
        Translates the labels from text to an id - from 1 to N, where N is the number of possible labels in the dataset
        """
        self.labels = list(set(self.dataset["train"]["answer"]))
        self.label2id = {}
        self.id2label = {}
        count = 0
        for label in self.labels:
            self.label2id[label] = count
            self.id2label[count] = label
            count += 1
        # print("label2id",label2id)
        # print("id2label", id2label)

    def prepareBatch(self, batch: dict) -> dict:
        """
        Prepares batch for model training. Sends batch to GPU. Returns the processed batch.

        Args:
            batch (dict): Batch given by torch.utils.data.DataLoader

        Returns:
            dict: processed batch in GPU, ready to be fed to model.
        """
        # create training batch
        img_paths = []
        #print("self.imagesPath", self.imagesPath)
        # RSVQAxBEN image folder is distributed across subfolders.
        if self.dataset_name == "RSVQAxBEN":
            batch["img_id"] = [os.path.join(str(img_id // 2000), str(img_id)) for img_id in batch["img_id"]]
        #print("batch imgs", batch["img_id"].tolist())

        for img_id in batch["img_id"].tolist():
            if os.path.exists(os.path.join(self.imagesPath, str(img_id) + ".jpg")):
                img_paths.append(os.path.join(self.imagesPath, str(img_id) + ".jpg"))
        #print("img_paths", img_paths)

        imgs_to_encode = [Image.open(img) for img in img_paths]

        # process the entire batch at once with padding for dynamic padding
        processed_batch = self.input_processor(
            text=batch["question"], images=imgs_to_encode, padding=True, return_tensors="pt")
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
        num_test_steps = len(self.test_loader)
        progress_bar = tqdm(range(num_test_steps))

        print("Computing metrics for test dataset.")
        metrics = datasets.load_metric("accuracy")
        self.model.eval()
        for batch in self.test_loader:
            batch = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            metrics.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar.update(1)

        if self.dataset_name == "RSVQA-HR":
            print("Computing metrics for test Philadelphia dataset.")
            num_test_steps = len(self.test_phili_loader)
            progress_bar = tqdm(range(num_test_steps))
            metrics_phili = datasets.load_metric("accuracy")
            for batch in self.test_phili_loader:
                batch = self.prepareBatch(batch)
                with torch.no_grad():
                    output = self.model(**batch)
                logits = output.logits
                predictions = torch.argmax(logits, dim=-1)
                metrics_phili.add_batch(predictions=predictions, references=batch["labels"])
                progress_bar.update(1)
            return metrics.compute(), metrics_phili.compute()

        return metrics.compute()

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
            batch = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            metrics.add_batch(predictions=predictions, references=batch["labels"])
        return metrics.compute(), output.loss.item()

    def trainPatience(self, epochs: dict, threshold: int = 3) -> bool:
        """
        Checks if the best model during training was achieved or not.

        Args:
            epochs (int): Number of epochs that already happened during training.
            threshold (int, optional): Threshold for the limit of epochs after the highest validation accuracy is achieved (patience). Defaults to 3.

        Returns:
            bool: Returns True if the limit for the training hasn't been reached yet.
        """
        # TODO Test this function
        print("all epochs:", epochs)
        highest_validation_epoch = self.getBestModel(epochs)
        print("highest validation epoch", highest_validation_epoch)
        return len(epochs) < highest_validation_epoch + threshold

    def getBestModel(epochs: dict) -> int:
        """
        Given a dictionary with the epochs data returns the epoch with the best model (highest validation metrics)

        Args:
            epochs (dict): Dictionary with data related to the training iteration (i.e. accuracies and loss)

        Returns:
            int: Epoch with the highest validation accuracy in the given epochs.
        """
        return max(epochs, lambda epoch: epochs[epoch]["validation metrics"]["accuracy"])

    def saveBestModel(self,  epochs: dict, folder_path: str) -> None:
        """
        Saves the model with the best performance.

        Args:
            epochs (dict): Dictionary with data related to all the training iterations (i.e. accuracies and loss for each epoch)
            folder_path (str): Folder inside logs folder where the model should be saved.
        """
        torch.save(epochs[self.getBestModel(epochs)]["model_state"], os.path(folder_path, "model.pth"))

    def saveTrain(self, log: dict, training_start_time: datetime.datetime) -> None:
        """
        Saves the complete log of the training, including the best model.

        Args:
            log (dict): Dictionary with all the information relevant to the training.
            training_start_time (datetime.datetime): Timestamp of when the training started
        """

        # create folder for the training session
        timestamp = str(training_start_time).split(".")[0]
        file_name = timestamp.replace(" ", "_").replace(":", "_").replace("-", "_")
        folder_path = os.path.join("logs", self.dataset_name, file_name)
        os.makedirs(folder_path)

        self.saveBestModel(log["epochs"])

        # delete the deep copy of the model's state dict from each epoch (already saved in its own file - model.pth inside the same folder as the log)
        for epoch in log["epochs"]:
            del epoch["model_state"]
        log["file name"] = file_name
        log["dataset"] = self.dataset_name
        log["device"] = torch.cuda.get_device_name(self.device)
        log["total elapsed time"] = str(datetime.datetime.now() - training_start_time)
        log["total epochs"] = len(log["epochs"])
        if self.dataset_name == "RSVQA-HR":
            log["test metrics"], log["test phili metrics"] = self.test()
        else:
            log["test metrics"] = self.test()

        with open(os.path.join(folder_path, file_name), "w") as logFile:
            json.dump(log, logFile, indent=4)

        print("Completed model training in", log["total elapsed time"], ".")

    def train(self) -> None:
        """
        Training loop.
        The training loop has two different stop conditions:
            1) a limit of epochs;
            2) if the interval of epochs between the highest validation accuracy and current epoch is higher than a threshold (default is 3).
        Once the training loop is complete a complete log is saved.
        """
        training_start_time = datetime.datetime.now()
        log = {}
        log["epochs"] = {}
        epochCount = 1

        # training loop
        while epochCount <= self.limit_epochs or self.trainPatience(log["epochs"], self.patience):
            epochStartTime = datetime.datetime.now()
            print(epochStartTime, "epoch", epochCount)
            epochProgress = tqdm(range(len(self.train_loader)))
            running_loss = 0.0
            train_accuracy = datasets.load_metric("accuracy")
            log["epochs"][epochCount] = {}

            for batch in self.train_loader:
                # encode batch and feed it to model
                batch = self.prepareBatch(batch)
                self.optimizer.zero_grad()

                output = self.model(**batch)
                # print("model output", output)
                output.loss.backward()
                running_loss += output.loss.item()
                self.optimizer.step()

                logits = output.logits

                predictions = torch.argmax(logits, dim=-1)
                train_accuracy.add_batch(predictions=predictions, references=batch["labels"])
                epochProgress.update(1)

            # current training loss and accuracy for each epoch
            log["epochs"][epochCount]["train metrics"] = train_accuracy.compute()
            log["epochs"][epochCount]["validation metrics"], log["epochs"][epochCount]["validation loss"] = self.validate()
            log["epochs"][epochCount]["train loss"] = running_loss/len(self.train_loader)
            log["epochs"][epochCount]["elapsed time"] = str(datetime.datetime.now - epochStartTime)
            log["epochs"][epochCount]["model_state"] = deepcopy(self.model.state_dict())
            epochCount += 1

        self.saveTrain(training_start_time)
