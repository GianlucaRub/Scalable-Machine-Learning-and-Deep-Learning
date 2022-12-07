# ID2223 Scalable Machine Learning and Deep Learning
## Lab2
**Professor:**
Jim Dowling

**Students:**

- Gianluca Ruberto
- Fabio Camerota

## Task description
- Fine tune a pre-trained [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) transformer model to transcribe Italian audio to text.
- Write a Gradio application that provides a User Interface to allow users to use your model in an useful or entertaining way.
- Explain how to improve the performance of your model with either model-centric improvements or data centric improvements
- Use the [Italian Common Voice Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/it/train) 11.0 from Mozilla Foundation.

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
