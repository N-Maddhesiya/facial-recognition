import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
OUTPUT_FILE = "encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Starting encoding...")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)
        print(f"  Processing: {image_path}")

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"  [WARN] No face found in {image_path}, skipping.")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)

        except Exception as e:
            print(f"  [ERROR] {image_path}: {e}")

data = {"encodings": known_encodings, "names": known_names}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"\n[DONE] Saved {len(known_names)} face(s) to '{OUTPUT_FILE}'")
print("Persons found:", set(known_names))