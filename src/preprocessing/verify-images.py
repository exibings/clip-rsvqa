import os


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
