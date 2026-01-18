from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form["message"]

        result = sentiment_analyzer(user_text)[0]
        label = result["label"]

        if label == "POSITIVE":
            recommendation = "У тебя отличное настроение!"
        elif label == "NEGATIVE":
            recommendation = "Ты немного грустишь."
        else:
            recommendation = "Нейтральное настроение."

    return render_template("index.html", recommendation=recommendation, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)
