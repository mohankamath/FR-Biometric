import os

folder = "data/processed/enrollment"
for person in os.listdir(folder):
    print(person)
    person_path = os.path.join(folder, person)
    for img in os.listdir(person_path):
        print("  ", img)
