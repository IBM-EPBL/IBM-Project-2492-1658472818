from flask import *
import os
from PIL import Image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename


app = Flask(__name__)

model = load_model('FV.h5')


labels = {0: 'apple', 1: 'banana', 2: 'bean', 3: 'beetroot', 4: 'bell pepper', 5: 'bitter gourd', 
          6: 'bottle gourd',7: 'broccoli', 8: 'cabbage', 9: 'capsicum', 10: 'carrot',
         11: 'cauliflower', 12: 'chicken', 13: 'chilli pepper', 14: 'corn', 15: 'cucumber',
         16: 'egg', 17: 'eggplant', 18: 'fish', 19: 'garlic', 20: 'ginger',
         21: 'grapes', 22: 'jalepeno', 23: 'kiwi', 24: 'lemon', 25: 'lettuce',
         26: 'mango', 27: 'mutton', 28: 'onion', 29: 'orange', 30: 'papaya',
         31: 'paprika', 32: 'pear', 33: 'peas', 34: 'pineapple', 35: 'pomegranate',
         36: 'potato', 37: 'pumpkin', 38: 'raddish', 39: 'soy beans', 40: 'spinach',
         41: 'sweetcorn', 42: 'sweetpotato', 43: 'tomato', 44: 'turnip', 45: 'watermelon'}

fruits = ['Apple','Banana','Bello Pepper','Chilli Pepper','Grapes','Jalepeno','Kiwi','Lemon','Mango','Orange','Paprika','Pear','Pineapple','Pomegranate','Watermelon','Papaya']
vegetables = ['Beetroot','Cabbage','Capsicum','Carrot','Cauliflower','Corn','Cucumber','Eggplant','Ginger','Lettuce','Onion','Peas','Potato','Raddish','Soy Beans','Spinach','Sweetcorn','Sweetpotato','Tomato','Turnip','Bean','Bitter Gourd','Bottle Gourd','Broccoli','Pumpkin']
non_vegetables=['Chicken', 'Egg', 'Fish', 'Mutton']



# original = ['adidas','alfaRomeo','Amazon','Apple','audi','bmw','chevrolet','citroen','Coca-Cola','dacia','Facebook','ferrari','fiat','ford','Google','honda','hyundai','jaguar','jeep','McDonald_s','NIKE','puma','starbucks']
# fake = ['fake-logo-adidas','fake-logo-apple','fake-logo-mcdonalds','fake-logo-nike','fake-logo-puma','fake-logo-starbucks']
# classes = ['adidas','alfaRomeo','Amazon','Apple','audi','bmw','chevrolet','citroen','Coca-Cola','dacia','Facebook','fake-logo-adidas','fake-logo-apple','fake-logo-mcdonalds','fake-logo-nike','fake-logo-puma','fake-logo-starbucks','ferrari','fiat','ford','Google','honda','hyundai','jaguar','jeep','McDonald_s','NIKE','puma','starbucks']
path = 'D:/IBM@2/'

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        print("Can't able to fetch the Calories")
        print(e)




def image_processing(img):
    imgpath = os.path.join(path,img)
    img = Image.open(imgpath).resize((250,250))
    img=load_img(imgpath,target_size=(224,224,3))
    try:
        img=img_to_array(img)
        img=img/255
        img=np.expand_dims(img,[0]) 

        # log_img=cv2.resize(log_img,(50,50))
        # image = np.array(log_img).flatten()
        # # data.append(image)
    except Exception as e:
        pass
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()



# def processed_img(img_path):
#     img = Image.open(img_file).resize((250,250))
#     img=load_img(img_path,target_size=(224,224,3))
#     img=img_to_array(img)
#     img=img/255
#     img=np.expand_dims(img,[0])
#     answer=model.predict(img)
#     y_class = answer.argmax(axis=-1)
#     print(y_class)
#     y = " ".join(str(x) for x in y_class)
#     y = int(y)
#     res = labels[y]
#     print(res)
#     return res.capitalize()


@app.route('/')
@app.route('/reg_form')
def reg_form():
    return render_template('reg_form.html')

@app.route('/form', methods = ['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route('/index', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/profile', methods = ['GET', 'POST'])
def profile():
    return render_template('profile.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        print(result)
        if result in vegetables:
                print('**Category : Vegetables**')
        elif result in non_vegetables:
                print('**Category : Non-Vegetables**')
        else:
            print('**Category : Fruit**')
            print("**Predicted : "+result+'**')

        cal = fetch_calories(result)
        print(cal)
        os.remove(file_path)
        return cal
    return None

if __name__ == '__main__':
    app.run(debug=True)