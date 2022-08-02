from flask import Flask,render_template,send_from_directory
from requests import request
import numpy as np
import pickle
import os
from flask import request


loaded_model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__,template_folder='templates',static_folder='static')

@app.route("/")

def home():

    return render_template("home.html")


#4route prediction
@app.route("/predict",methods=["get","post"])
def predict():
    
    pclass=request.form['pclass']
    sex=request.form['sex'] 
    age=request.form['age']
    sipsp=request.form['sipsp']
    parch=request.form['parch']
    ticket=request.form['ticket']
    fare=request.form['fare']
    cabin=request.form['cabin']
    embarked = request.form['embarked']
    
#take input from user form put in arry
    form_arry=np.array([[pclass,sex,age,sipsp,parch,ticket,fare,cabin,embarked]])
    #model.pickle.load(open("model.pkl","rb"))
    #predict on arry of user input form
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    prediction=loaded_model.predict(np.array(form_arry.astype(int)))
    classes = ["Dead","Survived"]
    result = classes[int(prediction)]  


    
    return render_template("result.html",result=result)

#2
if __name__ =="__main__":
    app.run(debug=True)