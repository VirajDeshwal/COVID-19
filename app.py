from flask import Flask, request, jsonify
from fastai.vision import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

# learner file
learn = load_learner(path="./models", file="covid-512-final.pkl")
classes = learn.data.classes


def image_predict(img_file):
    "image_predict will take image as an input and return prediction for COVID-19"
    prediction = learn.predict(open_image(img_file))
    result = prediction[2].numpy()
    return {
        "category": classes[prediction[1].item()],
        "result": {c: round(float(probs_list[i]), 3) for (i, c) in enumerate(classes)},
    }

# route
@app.route("/predict", methods=["POST"])
def predict():
    return jsonify(image_predict(request.files["image"]))


if __name__ == "__main__":
    app.run()
