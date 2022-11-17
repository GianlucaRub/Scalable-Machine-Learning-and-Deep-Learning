import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def passenger(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked):
    input_list = []
    if Pclass == "First Class":
        input_list.append(1)
    elif Pclass == "Second Class":
        input_list.append(2)
    else:
        input_list.append(3)
    input_list.append(Age)
    input_list.append(SibSp)
    input_list.append(Parch)
    input_list.append(Fare)
    if Sex == "Male":
        input_list.append(0)
        input_list.append(1)
    else:
        input_list.append(1)
        input_list.append(0)
    if Embarked == "Cherbourg":
        input_list.append(1)
        input_list.append(0)
        input_list.append(0)
    elif Embarked == "Queenstown":
        input_list.append(0)
        input_list.append(1)
        input_list.append(0)
    else:
        input_list.append(0)
        input_list.append(0)
        input_list.append(1)
    
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    res = str(res[0])
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    passenger_url = "https://raw.githubusercontent.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/main/Lab1/assets/" + res + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=passenger,
    title="Titanic Predictive Analytics",
    description="Insert passenger class, age, number of sibilings/spouse on board of the Titanic, number of parents/children on board of the Titanic, fare, sex, port of embarkation and see if he/she survived ",
    allow_flagging="never",
    inputs=[
        gr.inputs.Radio(choices=["First Class", "Second Class", "Third Class"], label="Passenger Class"),
        gr.inputs.Number(default=20, label="Age"),
        gr.inputs.Number(default=1.0, label="Number of sibilings/spouse on board of the Titanic"),
        gr.inputs.Number(default=1.0, label="Number of parents/children on board of the Titanic"),
        gr.inputs.Number(default=10.0, label="Fare"),
        gr.inputs.Radio(choices=["Male","Female"], label = "Sex"),
        gr.inputs.Radio(choices=["Cherbourg","Queenstown","Southampton"], label = "Port of embarkation")
        ],
    outputs=gr.Image(type="pil"))

demo.launch()

