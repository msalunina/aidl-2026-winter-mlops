import logging
import pathlib
import re

import torch
from flask import Flask, render_template, request

from model import SentimentAnalysis


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()


VOCAB_WORD2IDX = None
MODEL = None
NGRAMS = None

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

# The code in this function will be executed before we recieve any request
@app.before_first_request
def _load_model():
    # First load into memory the variables that we will need to predict
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
    checkpoint = torch.load(checkpoint_path)

    global VOCAB_WORD2IDX, MODEL, NGRAMS
    VOCAB_WORD2IDX = checkpoint["vocab_word2idx"]
    MODEL = SentimentAnalysis(len(VOCAB_WORD2IDX), checkpoint["embed_dim"], checkpoint["num_class"])
    MODEL.load_state_dict(checkpoint["model_state_dict"])

    NGRAMS = checkpoint["ngrams"]


# Disable gradients
@torch.no_grad()
def predict_review_sentiment(text):
    # Convert text to tensor using our tokenizer
    tokens = tokenize(text)
    # For now we only support unigrams (NGRAMS=1)
    text_tensor = torch.tensor(
        [VOCAB_WORD2IDX.get(token, VOCAB_WORD2IDX["<unk>"]) for token in tokens]
    )

    # Compute output
    if len(text_tensor) == 0:  # Handle empty input
        text_tensor = torch.tensor([VOCAB_WORD2IDX["<unk>"]])
    
    # For a single sequence, we need to keep it as 1D and provide the offset
    offsets = torch.tensor([0])
    output = MODEL(text_tensor, offsets)
    confidences = torch.softmax(output, dim=1)
    return confidences.squeeze()[
        1
    ].item()  # Class 1 corresponds to confidence of positive


@app.route("/predict", methods=["POST"])
def predict():
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """ Return an HTML. """
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
