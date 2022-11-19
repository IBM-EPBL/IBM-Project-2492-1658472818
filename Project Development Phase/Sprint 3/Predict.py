from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import requests
import json

app =  Flask(__name__, template_folder="templates")
model = load_model('FV.h5')
#print("Loaded model from disk")

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['GET', 'POST'])
def launch():
    if request.method == 'POST':
        f = request.form['ing']
        bp = os.path.dirname('__file__')
        fp = os.path.join(bp, "uploads", f.filename)
        f.save(fp)
        
        img = image.load_img(fp, target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)

        pred = np.argmax(model.predict(x), axis = 1)
        print("Prediction: ", pred)
        index = ['turnip','pumpkin','sweetcorn','raddish','ginger', 'lemon', 'pineapple','onion','potato','spinach','bean','jalepeno','orange', 'capsicum', 'banana',
 'peas', 'mutton', 'carrot', 'papaya' ,'beetroot', 'tomato', 'mango', 'sweetpotato', 'garlic', 'fish', 'eggplant' ,'cucumber', 'chicken',
 'bitter gourd', 'kiwi', 'paprika' ,'bottle gourd', 'corn', 'grapes', 'watermelon', 'pomegranate', 'broccoli', 'pear', 'bellpepper', 'chilli pepper',
 'egg', 'cabbage', 'lettuce', 'cauliflower', 'soy beans' ,'apple']

        result = str(index[pred[0]])

        f = open("Calories.json")
        data = json.load(f)
        cal = 0 
        for key, value in data.items():
            if key == result:
                cal = value
        
        q = request.form['quantity']
        totalCal = q*cal

        return render_template("predict.html", n = result, x = totalCal) 

if __name__ == "__main__":
    app.run(debug = True)