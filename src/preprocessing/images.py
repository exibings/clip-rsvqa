import os
from PIL import Image


def jpgOnly(datasetName) -> None:
    """
    Filters all dataset images to .jpg only.

    Args:
        datasetName (str): Dataset folder name
    """
    for img in os.listdir(os.path.join("datasets", datasetName, "images")):
        try:
            if not img.endswith(".jpg"):
                print("removing", img)
                os.remove(os.path.join("datasets", datasetName, "images", img))
        except:
            print("found the subfolder.")


def imageResizer(datasetName, imageSize=224) -> None:
    """
    Resizes all dataset images into a given image size.

    Args:
        datasetName (str): Dataset folder name
        imageSize (int, optional): Resized image sized. Defaults to 224.
    """
    if datasetName == "RSVQAxBEN":
        # total image checker
        images_checker = {}
        images_checker_resized = {}
        total = 0
        total_resized = 0
        for subfolder in os.listdir(os.path.join("datasets", "RSVQAxBEN", "images")):
            images_checker[subfolder] = 0
            images_checker[subfolder] = len(os.listdir(os.path.join("datasets", "RSVQAxBEN", "images", subfolder)))
            total += images_checker[subfolder] if subfolder != "resized" else 0
        for subfolder in os.listdir(os.path.join("datasets", "RSVQAxBEN", "images", "resized")):
            images_checker_resized[subfolder] = 0
            images_checker_resized[subfolder] = len(os.listdir(os.path.join(
                "datasets", "RSVQAxBEN", "images", "resized", subfolder)))
            total_resized += images_checker_resized[subfolder]

        print("total images", len(total), "distributed in", len(images_checker), "folders")
        print("already resized", len(total_resized), "distributed in", len(images_checker_resized), "folders")

    # image resizing for RSVQAxBEN
        # TODO ir buscar o código ao desktop. o ficheiro está na pasta do MBERT-VQA

    else:
        # total image checker
        img_list = os.path.join("datasets", datasetName, "images")
        print("total images", len(img_list))
        already_resized = os.path.join("datasets", datasetName, "images", "resized")
        print("already resized", len(already_resized))

        # image resizing for RSVQA-LR and RSVQA-HR
        for img in img_list:
            if img not in already_resized:
                if img.endswith(".jpg"):
                    print("resizing", img)
                    image = Image.open(os.path.join("datasets", datasetName, "images", img))
                    image = image.resize((imageSize, imageSize))
                    image.save(os.path.join("datasets", datasetName, "images", "resized", img))
