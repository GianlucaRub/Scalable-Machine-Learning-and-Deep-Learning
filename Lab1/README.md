# ID2223 Scalable Machine Learning and Deep Learning
## Lab1
**Professor:**
Jim Dowling

**Students:**

- Gianluca Ruberto
- Fabio Camerota

## Task description
- Write a feature pipeline that registers the titantic dataset as a Feature Group with [Hopsworks](https://www.hopsworks.ai/).
- Write a training pipeline that reads training data with a Feature View from Hopsworks, trains a binary classifier model to predict if a particular passenger survived the Titanic or not. Register the model with Hopsworks.
-  Write a Gradio application that downloads your model from Hopsworks and provides a User Interface to allow users to enter or select feature values to predict if a passenger with the provided features would survive or not.
- Write a synthetic data passenger generator and update your feature pipeline to allow it to add new synthetic passengers.
- Write a batch inference pipeline to predict if the synthetic passengers survived or not, and build a Gradio application to show the most recent synthetic passenger prediction and outcome, and a confusion matrix with historical prediction performance.
- Use the [Titanic Dataset](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv)
- Use [Modal](https://modal.com/) to run the code.

## Gradio Applications
- [Predict if a passenger would survive or not](https://huggingface.co/spaces/GIanlucaRub/Titanic)
- [Monitor the daily performance](https://huggingface.co/spaces/GIanlucaRub/Titanic-monitor)

## Serverless Infrastructure Model
![](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Lab1/assets/serverless_schema.png?raw=true)

## Feature Pipeline
The original dataset has been preprocessed in the following way:
- Columns PassengerId, Name and ticket has been removed since they were too specific and could lead to overfitting.
- Column Cabin has been removed since it had too many null values (77%).
- Columns Sex and Embarked has been one hot encoded.

After the preprocessing, the feature group has been created on Hopsworks.

## Training Pipeline
- Gradient Boosting Classifier from Keras has been selected as model to use.
- The feature view given as input is made entirely from the previously created feature group.
- The model has been saved in Hopsworks' model registry.
- The script runs on Modal daily.
- The dataset has been split in train and test with a 80/20 ratio.
- The results of the evaluation on the test set are:

![](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Lab1/titanic_model/confusion_matrix.png?raw=true)

## Daily Feature Pipeline for Synthetic Data
- Data has been generate by taking random values from each column of the original cleaned dataset.
- There is a 50/50 probability for the synthetic passenger to survive or not.
- Data generated is then added to Hopsworks' feature group, on which the model is trained daily.
- The script runs on Modal daily.

## Batch Inference Pipeline
- Every day the script runs on Modal and evaluates the last passenger added in the dataset.
- Results in the form of a confusion matrix and a table with the last 4 predictions are saved on Hopsworks in order to be accessed from the [Gradio Monitor application](https://huggingface.co/spaces/GIanlucaRub/Titanic-monitor).
