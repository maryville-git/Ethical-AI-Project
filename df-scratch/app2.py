from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class DeepfakeDetector:
    def __init__(self):
        
        self.model_name = "model/deepfake_detector/checkpoint-34" 
        try:
            print("Loading model...")
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            print("Model loaded successfully.")

            print("Loading feature extractor...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            print("Feature extractor loaded successfully.")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model or feature extractor: {e}")

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to open image: {e}"}

        try:
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=-1)

            predicted_label = predictions.argmax().item()
            confidence = predictions[0][predicted_label].item()

            return {
                "is_fake": bool(predicted_label),
                "confidence": confidence,
                "prediction": "Fake" if predicted_label else "Real",
            }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

detector = DeepfakeDetector()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return redirect(url_for("home"))  # Redirect to home page if accessed directly

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            result = detector.predict_image(filepath)
            result["confidence"] = f"{result['confidence']:.2%}"
            result["image_path"] = os.path.join("uploads", filename)
            return render_template("result.html", result=result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return (
            jsonify(
                {
                    "error": "File type not allowed. Please upload JPG or PNG images only."
                }
            ),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
