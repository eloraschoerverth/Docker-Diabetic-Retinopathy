
import os
import csv
from fastai.vision import *

# Import Flask for creating API
from flask import Flask, request



# Load the trained model from current directory
# defaults.device = torch.device('cpu')
print("Defaults for device set to CPU " )

def predictgen(learnmodel,img):
    pred_class,pred_idx,outputs = learnmodel.predict(img)
    class_sorted = pred_class.obj
    return class_sorted

learnretino = load_learner(path="./model", file="model.pkl")
print("Diabetic Retinopathy Model loaded ")

# Initialise a Flask app
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

# POST endpoint receives labeled csv and images

@app.route("/predict")

def predict_retino():

    accuracy = 0
    cntImg = 128
    f=open('valid-aidr-metadata-updated.csv')
    first = True

    for row in csv.reader(f):
        if first:
            first= False
            continue
        print("Processing Image:"+row[8])
        try:
            image_fastai = open_image('ImageData/'+row[8])
            class_diag  = predictgen(learnretino,image_fastai)
            diag_names  = {"nongradable" : "Nongradable", "normal" :  "Normal", "retinopathy" : "Retinopathy" }
            if row[6] == diag_names[class_diag].lower():
                accuracy +=1
        except IOError:
            cntImg-=1
            pass


    # return the accuracy

    return "The accuracy of the model is " + str(round(accuracy/cntImg,3))

if __name__ == "__main__":

    app.run(debug=True,host="0.0.0.0",port=port)
