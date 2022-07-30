import os

from PIL import Image


def jpgOnly(dataset_name: str) -> None:
    """
    Filters all dataset images to .jpg only.

    Args:
        dataset_name (str): Dataset folder name
    """
    if dataset_name == "RSVQAxBEN":
        for subfolder in os.listdir(os.path.join("datasets", dataset_name, "images")):
            for file in os.listdir(os.path.join("datasets", dataset_name, "images", subfolder)):
                try:
                    if not file.endswith(".jpg"):
                        print("removing", file)
                        os.remove(os.path.join("datasets", dataset_name, "images", file))
                except:
                        print("found the \"resized\" subfolder.")
    else:
        for file in os.listdir(os.path.join("datasets", dataset_name, "images")):
            try:
                if not file.endswith(".jpg"):
                    print("removing", file)
                    os.remove(os.path.join("datasets", dataset_name, "images", file))
            except:
                print("found the \"resized\" subfolder.")


def imageResizer(dataset_name: str, imageSize: int = 224) -> None:
    """
    Resizes all dataset images into a given image size.

    Args:
        dataset_name (str): Dataset folder name
        imageSize (int, optional): Resized image sized. Defaults to 224.
    """
    if dataset_name == "RSVQAxBEN":
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
        for subfolder in images_checker:
            if subfolder not in images_checker_resized:
                os.makedirs(os.path.join("datasets", "RSVQAxBEN", "images", "resized", subfolder))
            for img in images_checker[subfolder]:
                if img.endswith(".jpg") and not os.path.exists(os.path.join("datasets", "RSVQAxBEN", "images", "resized", subfolder, img)):
                    print("resizing", img)
                    image = Image.open(os.path.join("datasets", "RSVQAxBEN", "images", subfolder, img))
                    image = image.resize((imageSize, imageSize))
                    image.save(os.path.join("datasets", "RSVQAxBEN", "images", "resized", subfolder, img))
    else:
        # total image checker
        img_list = os.listdir(os.path.join("datasets", dataset_name, "images"))
        print("total images", len(img_list))
        already_resized = os.listdir(os.path.join("datasets", dataset_name, "images", "resized"))
        print("already resized", len(already_resized))

        # image resizing for RSVQA-LR and RSVQA-HR
        for img in img_list:
            if img not in already_resized:
                if img.endswith(".jpg"):
                    print("resizing", img)
                    image = Image.open(os.path.join("datasets", dataset_name, "images", img))
                    image = image.resize((imageSize, imageSize))
                    image.save(os.path.join("datasets", dataset_name, "images", "resized", img))



def verifyImages():
    print("Verifying RSVQA-LR images...")
    print("\tall original images?", True if len(os.listdir(os.path.join("datasets", "RSVQA-LR", "images"))) == 772+1 else False)
    print("\tall resized images?", True if len(os.listdir(os.path.join(
    "datasets", "RSVQA-LR", "images", "resized"))) == 772 else False)

    print("Verifying RSVQA-HR images...")
    print("\tall original images?", True if len(os.listdir(os.path.join("datasets", "RSVQA-HR", "images"))) == 10659+1 else False)
    print("\tall resized images?", True if len(os.listdir(os.path.join(
        "datasets", "RSVQA-LR", "images", "resized"))) == 10659 else False)


    print("Verifying RSVQAxBEN images...")
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

    # print("original:", json.dumps(images_checker, indent=2), "total original:", total)
    print("\tall original images?", True if total == 590326 else False)
    # print("resized", json.dumps(images_checker_resized, indent=2), "total resized", total_resized)
    print("\tall resized images?", True if total == 590326 else False)
