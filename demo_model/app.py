import os
import csv
# from fastai.vision import *

# Import Flask for creating API
from flask import Flask, request

# Initialise a Flask app
app = Flask(__name__)

# Load the trained model from current directory
# defaults.device = torch.device('cpu')
# print("Defaults for device set to CPU " )
#
# def predictgen(learnmodel,img):
#     pred_class,pred_idx,outputs = learnmodel.predict(img)
#     class_sorted = pred_class.obj
#     return class_sorted


@app.route('/')
def ping():
    return 'ok'


@app.route("/predict")
def predict(images):
    #generate empty csv file to store predictions
    f = open('predictions.csv','w')

    learnretino = load_learner(path="./model", file="model.pkl")
    print("Diabetic Retinopathy Model loaded ")

    filelist=os.listdir('images')

    for imgfile in filelist:

        image_fastai = open_image('images/'+imgfile)
        class_diag  = predictgen(learnretino,image_fastai)
        diag_names  = {"nongradable" : "Nongradable", "normal" :  "Normal", "retinopathy" : "Retinopathy" }

        writer = csv.writer(f)
        writer.writerow(imgfile,diag_names[class_diag])

    f.close()

    return f

if __name__ == '__main__':
   app.run(host='0.0.0.0',port=5000)
