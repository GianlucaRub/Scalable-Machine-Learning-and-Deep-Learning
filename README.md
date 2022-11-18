# ID2223 Scalable Machine Learning and Deep Learning
## Lab1
**Professor:** Jim Dowling
**Students:**
-Gianluca Ruberto
-Fabio Camerota

## Task description
- Write a feature pipeline that registers the titantic dataset as a Feature Group with Hopsworks.
- Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a binary classifier model to predict if a particular passenger survived the Titanic or not. Register the model with Hopsworks.
-  Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
- Write a synthetic data passenger generator and update your feature pipeline to allow it to add new synthetic passengers.
- Write a batch inference pipeline to predict if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
- Use The Titanic Dataset: https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv

## Serverless Infrastructure Model
![](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Lab1/assets/serverless_schema.png?raw=true)

# Feature Pipeline
The original dataset has been preprocessed in the following way:
- Columns PassengerId, Name and ticket has been removed since they were too specific and could lead to overfitting.
- Column Cabin has been removed since it had too many null values (77%)
- Columns Sex and Embarked has been one hot encoded

After the preprocessing, the feature group has been created on Hopsworks.
