from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle
import itertools


app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    x_new = []

    scaler_path=os.path.join('C:/Users/asus/Desktop/VI SEM/Minor project/stroke prediction/','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    gen = []
    job = []
    sm = []
    if(gender==0):
        gen = [1,0,0]
    elif(gender==1):
        gen = [0,1,0]
    else:
        gen = [0,0,1]

    if(work_type==0):
        job=[1,0,0,0,0]
    elif(work_type==1):
        job=[0,1,0,0,0]
    elif(work_type==2):
        job=[0,0,1,0,0]
    elif(work_type==3):
        job=[0,0,0,1,0]
    else:
        job=[0,0,0,0,1]

    if(smoking_status==0):
        sm=[1,0,0,0]
    elif(smoking_status==1):
        sm=[0,1,0,0]
    elif(smoking_status==2):
        sm=[0,0,1,0]
    else:
        sm=[0,0,0,1]

    x_new.append(gen)
    x_new.append(job)
    x_new.append(sm)
    x_new.append([age,hypertension,heart_disease,ever_married,Residence_type,avg_glucose_level,bmi])

    x_final = list(itertools.chain(*x_new))
    x_final = np.array(x_final).reshape(1,-1)


    x_final=scaler.transform(x_final)


    model_path=os.path.join(r'C:\Users\ROHAN\Downloads\project\stroke prediction\models\rf.sav')
    rf=joblib.load(model_path)

    Y_pred=rf.predict(x_final)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
    app.run(debug=True,port=7384)
