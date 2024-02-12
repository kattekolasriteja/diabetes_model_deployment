# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:18:35 2024

@author: Ketan
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app=FastAPI()

class model_input(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
    
    
model_path = 'trained_model.sav'

# Load the model
loaded_model = pickle.load(open(model_path, 'rb'))


@app.post('/diabetes_prediction')


def diabetes_pred(input_parameters: model_input):
    input_data=input_parameters.json()
    input_dict=json.loads(input_data)
    preg=input_dict['Pregnancies']
    glu=input_dict['Glucose']
    bp=input_dict['BloodPressure']
    skin=input_dict['SkinThickness']
    insulin=input_dict['Insulin']
    bmi=input_dict['BMI']
    dpf=input_dict['DiabetesPedigreeFunction']
    age=input_dict['Age']
    input_list=[preg,glu,bp,skin,insulin,bmi,dpf,age]
    pred=loaded_model.predict([input_list])
    if pred[0]==0:
        return 'The person is not Diabetic'
    else:
        return 'The person is Diabetic'