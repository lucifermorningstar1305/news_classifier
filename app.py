from flask import Flask, request, render_template
import json
import os

from predict import predict

app = Flask(__name__)

@app.route("/")
def hello():

	return json.dumps({"message":"Hello World", "statusCode":200})


@app.route("/predict", methods=["GET"])
def predict_news():
	text = request.args.get('text')

	prediction, prob = predict(text)

	return json.dumps({"message":f"This is a {prediction} news", "probability":prob, "statusCode":200})

@app.route("/categorize")
def categorize_news():
	return render_template('predict.html')

@app.route("/predict_data", methods=["POST"])
def predict_data():
	news = request.form['news']
	prediction, prob = predict(news)

	return json.dumps({"news":news, "message": f'This news belongs to {prediction}', "probability":prob, "statusCode":200})

if __name__ == "__main__":

	port = int(os.environ.get("PORT", 5000))

	app.run(host="0.0.0.0", port=port)
