import datetime
import json
import os
from copy import deepcopy

import datasets
import torch
from PIL import Image
from torch.optim import optim
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from Model import CLIPxRSVQA


class Trainer:
    def __init__(self, limitEpochs=25, batchSize=64, useResizedImages=False, datasetName=None) -> None:
        self.limitEpochs = limitEpochs
        self.batchSize = batchSize

        self.imagesPath = os.path.join("datasets", datasetName, "images") if not useResizedImages else os.path.join(
            "datasets", datasetName, "images", "resized")
        self.datasetName = datasetName
        self.dataset = datasets.load_from_disk(os.path.join("datasets", datasetName, "dataset"))
        self.encodeDatasetLabels()
        self.trainLoader = torch.utils.data.DataLoader(self.dataset["train"], batch_size=self.batchSize,
                                                       shuffle=True, num_workers=2)

        self.testLoader = torch.utils.data.DataLoader(self.dataset["test"], batch_size=self.batchSize,
                                                      shuffle=False, num_workers=2)
        if datasetName == "RSVQA-HR":
            self.testPhiliLoader = torch.utils.data.DataLoader(self.dataset["test_phili"], batch_size=self.batchSize,
                                                               shuffle=False, num_workers=2)
        self.validationLoader = torch.utils.data.DataLoader(self.dataset["validation"], batch_size=self.batchSize,
                                                            shuffle=False, num_workers=2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputProcessor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")

        clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")

        self.model = CLIPxRSVQA(clip_model.config, num_labels=len(self.label2id))
        self.model.text_model = clip_model.text_model
        self.model.vision_model = clip_model.vision_model
        self.model.visual_projection = clip_model.visual_projection
        self.model.text_projection = clip_model.text_projection
        self.model.logit_scale = clip_model.logit_scale

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

        # TODO criar verificacoes para nao abusar das GPUs todas
        self.model.to(self.device)  # send model to GPU

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
        #print("id2label", id2label)

    def prepareBatch(self, batch) -> dict:
        """ 
        Prepares batch for model training. Sends batch to GPU. Returns the processed batch.

        Args:
            batch (dict): Batch given by torch.utils.data.DataLoader

        Returns:
            dict: processed batch in GPU, ready to be fed to model.
        """
        # TODO ver qual é o tipo exato do batch que é passado ao prepareBatch (fazer print de type(batch))
        # create training batch
        imgNames = []
        for img_id in batch["img_id"].tolist():
            if str(img_id) + ".jpg" in self.imagesPath:
                if self.datasetName == "RSVQAxBEN":
                    # RSVQAxBEN image folder is distributed across subfolders.
                    imgNames.append(os.path.join(str(img_id // 2000), str(img_id) + ".jpg"))
                else:
                    imgNames.append(str(img_id) + ".jpg")
        imgs_to_encode = [Image.open(os.path.join(self.imagesPath, img)) for img in imgNames]

        # process the entire batch at once with padding for dynamic padding
        processed_batch = self.processor(
            text=batch["question"], images=imgs_to_encode, padding=True, return_tensors="pt")
        del imgs_to_encode  # free up memory from imgs
        processed_input = {**{"labels": torch.tensor([self.label2id[label]
                                                     for label in batch["answer"]])}, **dict(processed_batch)}

        # send tensors to GPU
        for key in processed_input:
            processed_input[key] = processed_input[key].to(self.device)
        return processed_input

    def test(self, testLoader) -> float:
        """
        Evaluates the trainer's model performance (accuracy) with the given test dataset.  

        Args:
            testLoader (torch.utils.data.DataLoader): Supplies the batches to be processed from the test dataset.

        Returns:
            dict: Dictionary with the performance metrics (accuracy) for the trainer's model with the test dataset.
        """
        num_test_steps = len(testLoader)
        progress_bar = tqdm(range(num_test_steps))

        accuracy = datasets.load_metric("accuracy")
        self.model.eval()
        for batch in testLoader:
            batch = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar.update(1)

        return accuracy.compute()["accuracy"]

    def validate(self, validationLoader):
        """
        Evaluates the trainer's model performance (accuracy) with the given validation dataset.  

        Args:
            validationLoader (torch.utils.data.DataLoader): Supplies the batches to be processed from the validation dataset.

        Returns:
            dict: Dictionary with the performance metrics (accuracy) for the trainer's model with the validation dataset.
        """
        accuracy = datasets.load_metric("accuracy")
        self.model.eval()
        for batch in validationLoader:
            batch = self.prepareBatch(batch)
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy.add_batch(predictions=predictions, references=batch["labels"])
        return accuracy.compute()["accuracy"], output.loss.item()

    def verifyTrainStop(self, epochs, threshold=3) -> bool:
        """
        Checks if the best model during training was achieved or not.

        Args:
            epochs (int): Number of epochs that already happened during training.
            threshold (int, optional): Threshold for the limit of epochs after the highest validation accuracy is achieved. Defaults to 3.

        Returns:
            bool: Returns True if the limit for the training hasn't been reached yet.
        """
        # TODO Test this function
        print("all epochs:", epochs)
        highestValidationEpoch = self.getBestModel(epochs)
        print("highest validation epoch", highestValidationEpoch)
        return len(epochs) < highestValidationEpoch + threshold

    def getBestModel(epochs) -> int:
        """
        Given a dictionary with the epochs data returns the epoch with the best model (highest validation accuracy)

        Args:
            epochs (dict): Dictionary with data related to the training iteration (i.e. accuracies and loss)

        Returns:
            int: Epoch with the highest validation accuracy in the given epochs.
        """
        return max(epochs, lambda epoch: epochs[epoch]["validation accuracy"])

    def saveBestModel(self,  epochs, folderPath) -> None:
        """
        Saves the model with the best performance.

        Args:
            epochs (dict): Dictionary with data related to all the training iterations (i.e. accuracies and loss for each epoch)
            folderPath (str): Folder inside logs folder where the model should be saved.
        """
        torch.save(epochs[self.getBestModel(epochs)]["model_state"], os.path(folderPath, "model.pth"))

    def saveTrain(self, log, trainingStartTime) -> None:
        """
        Saves the complete log of the training, including the best model.

        Args:
            log (dict): Dictionary with all the information relevant to the training.
            trainingStartTime (datetime.datetime): Timestamp of when the training started
        """

        # create folder for the training session
        timestamp = str(trainingStartTime).split(".")[0]
        fileName = timestamp.replace(" ", "_").replace(":", "_").replace("-", "_")
        folderPath = os.path.join("logs", self.datasetName, fileName)
        os.makedirs(folderPath)

        self.saveBestModel(log["epochs"])

        # delete the deep copy of the model's state dict from each epoch (already saved in its own file - model.pth inside the same folder as the log)
        for epoch in log["epochs"]:
            del epoch["model_state"]
        log["file name"] = fileName
        log["dataset"] = self.datasetName
        log["device"] = torch.cuda.get_device_name(self.device)
        log["total elapsed time"] = str(datetime.datetime.now() - trainingStartTime)
        log["total epochs"] = len(log["epochs"])
        log["test accuracy"] = self.test(self.testLoader)
        with open(os.path.join(folderPath, fileName), "w") as logFile:
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
        trainingStartTime = datetime.datetime.now()
        log = {}
        epochCount = 1

        # training loop
        while epochCount <= self.limitEpochs or self.verifyTrainStop(log["epochs"]):
            epochStartTime = datetime.datetime.now()
            epochProgress = tqdm(range(len(self.trainLoader)))
            running_loss = 0.0
            trainAccuracy = datasets.load_metric("accuracy")
            log["epochs"][epochCount] = {}

            for batch in self.trainLoader:
                # encode batch and feed it to model
                batch = self.prepareBatch(batch)
                self.optimizer.zero_grad()

                output = self.model(**batch)
                #print("model output", output)
                output.loss.backward()
                running_loss += output.loss.item()
                self.optimizer.step()

                logits = output.logits

                predictions = torch.argmax(logits, dim=-1)
                trainAccuracy.add_batch(predictions=predictions, references=batch["labels"])
                epochProgress.update(1)

            # current training loss and accuracy for each epoch
            log["epochs"][epochCount]["train accuracy"] = trainAccuracy.compute()["accuracy"]
            log["epochs"][epochCount]["validation accuracy"], log["epochs"][epochCount]["validation loss"] = self.validate(
                self.validationLoader)
            log["epochs"][epochCount]["train loss"] = running_loss/len(self.trainLoader)
            log["epochs"][epochCount]["elapsed time"] = str(datetime.datetime.now - epochStartTime)
            log["epochs"][epochCount]["model_state"] = deepcopy(self.model.state_dict())
            epochCount += 1

        self.saveTrain(trainingStartTime)
