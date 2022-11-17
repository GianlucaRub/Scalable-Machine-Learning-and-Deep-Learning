import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(survived):
    """
    Returns a single Titanic passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random
    
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    # removing columns not useful for prediction
    titanic_df.drop(["PassengerId","Cabin","Name","Ticket","Survived"],inplace=True,axis = 1)
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
    
    length = titanic_df.shape[0] - 1
    
    

    df = pd.DataFrame({"Survived": [survived]})
    # generating value from the dataset for not one hot encoded columns
    columns = ["Pclass", "Age", "SibSp","Parch","Fare"]
    for column in columns:
        row = random.randint(0,length)
        temp_list = list(titanic_df[column])
        df[column] = temp_list[row]
        
    # generating random values for the one hot encoded features
    row = random.randint(0,length)
    sex_cols = ["Sex_female","Sex_male"]
    for i in range(2):
        temp_list = list(titanic_df[sex_cols[i]])
        df[sex_cols[i]] = temp_list[row]

    row = random.randint(0,length)
    embarked = random.randint(0,2)
    embarked_cols = ["Embarked_C","Embarked_Q","Embarked_S"]
    for i in range(3):
        temp_list = list(titanic_df[embarked_cols[i]])
        df[embarked_cols[i]] = temp_list[row]
	
    # setting to int32 since they are int and not bigint
    encoded_cols = sex_cols + embarked_cols
    df[encoded_cols] = df[encoded_cols].astype("int32")
    return df

def get_random_titanic_passenger():
    """
    Returns a DataFrame containing one random Titanic passenger
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(1)
    not_survived_df = generate_passenger(0)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.randint(0,1)
    if pick_random ==1:
        titanic_df = survived_df
        print("Survivor added")
    else:
        titanic_df = not_survived_df
        print("Not survivor added")

    return titanic_df

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_titanic_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
