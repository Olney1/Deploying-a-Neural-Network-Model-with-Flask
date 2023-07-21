from flask import Flask, request, render_template
from fastai.vision.all import load_learner, PILImage
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load the learner
learn = load_learner('human_dog_pigeon_baby_model_v2.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    error = None
    prediction = None
    confidence = None
    image_b64 = None

    if request.method == 'POST':
        # get the file from the post request
        file = request.files['file']

        # if no file is selected
        if file.filename == '':
            error = 'No selected file.'
        else:
            # check file extension
            filename, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in ['.jpg', '.png', '.jpeg']:
                error = 'Invalid file type. We only accept jpg, png or jpeg image files.'
            else:
                img = PILImage.create(file.read())
                pred_class, pred_idx, outputs = learn.predict(img)
                
                confidence_score = outputs[pred_idx.item()].item()
                # Here it is useful to show the confidence score as we test the app with different images. Comment this line out when happy.
                print(confidence_score)

                # Here we set the confidence score in order to determine the cut-off point of confidence for the prediction to default to 'Unknown' for images the model has not been trained on or not confident about.
                if confidence_score < 0.97:
                    prediction = 'Unknown'
                else:
                    prediction = str(pred_class)

                # Here we modify the prediction if it's equal to 'babies' just because that was how the model was trained.
                if prediction == 'babies':
                    prediction = 'Baby'
                else:
                    prediction = prediction.capitalize()
                
                confidence = "{:.2%}".format(confidence_score)

                # convert image into a base64 format
                buf = BytesIO()
                img.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode()
            
    # Render form again and add prediction results
    return render_template('index.html', prediction=prediction, confidence=confidence, error=error, image=image_b64)

if __name__ == '__main__':
    app.run()
