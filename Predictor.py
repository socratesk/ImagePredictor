# import the Flask class from the flask module
from flask import Flask, render_template, request, jsonify

# REFERENCE: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html?__s=kiszvfzpzeidxjqjknxq

from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Create an application object
app = Flask(__name__)

# Render image upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Execute prediction function
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
	data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
	if request.method == "POST":

		print (request.files)
		if request.files.get("file"):

			# read the image in PIL format
			image = request.files["file"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			with graph.as_default():
				preds = model.predict(image)
				results = imagenet_utils.decode_predictions(preds)
				data["predictions"] = []

				# loop over the results and add them to the list of returned predictions
				for (imagenetID, label, prob) in results[0]:
					r = {"label": label, "probability": float(prob)*100}
					data["predictions"].append(r)

				# indicate that the request was a success
				data["success"] = True

				if data["success"]:
					predStr = ""
					# loop over the predictions and display them
					for (i, result) in enumerate(data["predictions"]):
						#print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
						tempStr = "\n{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"])
						#print (tempStr)
						predStr = predStr + tempStr

					print (predStr)
					data["predStr"] = predStr

				# otherwise, the request failed
				else:
					print("Request failed")

		# return the data dictionary as a JSON response
		return (jsonify(data))



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")
	global graph
	graph = tf.get_default_graph()


# use decorators to link the function to a url
@app.route('/')
def home():
    return "Hello, World!"  # return a string


# start the server with the 'run()' method
if __name__ == "__main__":
	print(("*Starting server... \n Please wait until server has fully started"))
	load_model()
	app.run()
