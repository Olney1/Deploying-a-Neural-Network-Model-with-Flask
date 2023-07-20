from flask import Flask, request, render_template
from fastai.vision.all import load_learner, PILImage
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load the learner
learn = load_learner('human_dog_pigeon_baby_model.pkl')

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
                pred_class,pred_idx,outputs = learn.predict(img)
                prediction = str(pred_class)
                confidence_raw_format = str(outputs[pred_idx.item()])
                confidence = "{:.2%}".format(outputs[pred_idx.item()])

                # Here we modify the prediction if it's equal to 'babies'
                if prediction == 'babies':
                    prediction = 'Baby'
                if prediction == 'pigeon':
                    prediction = 'Pigeon'
                if prediction == 'human':
                    prediction = 'Human'
                if prediction == 'dog':
                    prediction = 'Dog'

                # convert image into a base64 format
                buf = BytesIO()
                img.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode()
            
    # Render form again and add prediction results
    return render_template('index.html', prediction=prediction, confidence=confidence, error=error, image=image_b64)

if __name__ == '__main__':
    app.run()
