# ID2223 Scalable Machine Learning and Deep Learning
## Lab2
**Professor:**
Jim Dowling

**Students:**

- Gianluca Ruberto
- Fabio Camerota

## Task description
- Fine tune a pre-trained [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) transformer model to transcribe Italian audio to text;
- Write a Gradio application that provides a User Interface to allow users to use your model in an useful or entertaining way;
- Explain how to improve the performance of your model with either model-centric improvements or data centric improvements;
- Use the [Italian Common Voice Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/it/train) 11.0 from Mozilla Foundation.
- Use [Google Colab](https://colab.research.google.com/)
## Gradio Applications
[WebApp](https://huggingface.co/spaces/GIanlucaRub/whisper-it) that allows you to:
- Upload and transcribe an audio;
- Record and transcribe an audio;
- Upload and transcribe a video;
- Transcribe a youtube video.


## Feature Pipeline
The feature pipeline has been heavily affected by two aspects: heavy limitations in the resource provided by colab, despite having paid a Colab Pro subscription, and due to the huge size of the complete preprocessed dataset, around 200 GB. As result, we decided to use as training set just the first 10% of the complete train and validation data. The evaluation (or validation) is composed by the first 10% of the complete test data.
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
