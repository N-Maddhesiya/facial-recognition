from flask import Flask, request, jsonify, render_template
import face_recognition
import numpy as np
import pickle
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load pickle file once at startup
print("[INFO] Loading encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"[INFO] Loaded {len(known_names)} face(s): {set(known_names)}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    payload = request.get_json()
    if not payload or "image" not in payload:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image from browser
    img_data = payload["image"].split(",")[1]  # strip "data:image/jpeg;base64,"
    img_bytes = base64.b64decode(img_data)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = np.array(pil_image)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    results = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        confidence = 0.0

        if len(face_distances) > 0:
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = known_names[best_match_idx]
                confidence = round((1 - face_distances[best_match_idx]) * 100, 1)

        results.append({
            "name": name,
            "confidence": confidence,
            "box": {"top": top, "right": right, "bottom": bottom, "left": left}
        })

    return jsonify({"faces": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)