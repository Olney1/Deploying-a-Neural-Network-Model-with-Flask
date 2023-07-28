# Deploying a Neural Network Model with Flask

This project demonstrates how to deploy a FastAI-based image classification model into a web application using Flask. The web application classifies uploaded images into one of four categories: Pigeon, Dog, Adult, and Baby.

## About the Project

The primary goal of this project is to show the process of integrating a machine learning model into a Flask application. The model is a convolutional neural network (CNN) trained with the FastAI library. Flask, a lightweight web framework for Python, is used to create an interface between the user and the model.

## How to Use

1. Clone the repository: `git clone https://github.com/Olney1/Image-Classification.git`
2. Navigate to the project directory: `cd Image-Classification`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the Flask application: `python app.py`
5. Open your web browser and visit `localhost:5000` to see the application in action.

## How it Works

The user uploads an image through the web interface. The image is then processed and fed into the neural network model. The model makes a prediction about the class of the image (Pigeon, Dog, Adult, Baby, or Unknown if the confidence score is below a certain threshold). The prediction, along with the associated confidence score, is then displayed on the webpage.

## Contributing

This project serves as a starting point and can be expanded to suit other use-cases. If you'd like to contribute, please feel free to make a pull request.

## Note

The current model is specifically trained to classify images into four categories. For more diverse or accurate results, the model should be retrained on a different dataset.
