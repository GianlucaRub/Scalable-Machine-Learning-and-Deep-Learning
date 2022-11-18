import os
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
# removing columns not useful for prediction
titanic_df.drop(["PassengerId","Cabin","Name","Ticket"],inplace=True,axis = 1)
# removing rows with null values
titanic_df = titanic_df.dropna()
# one hot encoding categorical variables
sex = pd.get_dummies(titanic_df.Sex, prefix="Sex")
titanic_df[sex.columns] = sex
titanic_df.drop(["Sex"],inplace=True,axis = 1)
embarked = pd.get_dummies(titanic_df.Embarked, prefix="Embarked")
embarked.columns
titanic_df[embarked.columns] = embarked
titanic_df.drop(["Embarked"],inplace=True,axis = 1)

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=["Pclass","Age","SibSp","Parch","Fare","Sex_male","Sex_female","Embarked_C","Embarked_Q","Embarked_S"], 
    description="Titanic dataset")
titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

