from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
model=load_model("./Model/digitrec.h5")
fileno=1
app=Flask(__name__,template_folder='template',static_folder='./static')
def predictRes():
    global model
    img=Image.open("./static/result.png").convert("L")
    img=img.resize((28,28))
    im2arr=np.array(img)
    im2arr=im2arr.reshape(1,28,28,1)
    y_pred=model.predict(im2arr)
    re=list(y_pred[0]).index(max(y_pred[0]))
    plt.bar(list(range(10)),y_pred[0],align="center")
    plt.xticks(list(range(10)),list(range(10)) )
    plt.xlabel("Digits")
    plt.ylabel("Accuracy")
    plt.savefig('./static/graph.png')
    plt.clf()
    return re
    
   
@app.route('/')
def home():
   return render_template('./index.html')
@app.route('/app')
def main():
   return render_template('./web.html',showcase="1")
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("./static/result.png")
      res=predictRes()
      return render_template('./result.html',showcase=str(res))
if __name__ == '__main__':
   app.run()