import os
import modal
#import great_expectations as ge
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
titanic_df.drop(["PassengerId","Cabin","Name","Ticket"],inplace=True,axis = 1)
titanic_df = titanic_df.dropna()

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"], 
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

