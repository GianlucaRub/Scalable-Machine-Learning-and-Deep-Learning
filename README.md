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
Steps followed in the feature pipeline:
- Download the dataset;
- Remove not useful features such as accent, age, gender, etc, leaving just the audio array;
- Change the sample rate to 48000 to 16000;
- Apply [Whisper Feature Extractor Tiny](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperFeatureExtractor);
- Apply [Whisper Tokenizer Tiny](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperTokenizer).

## Augmented Feature Pipeline
In order to improve the performance, data augmentation techniques have been applied to the train data, including:
- Adding random noise;
- Pitch modification;
- Speed modification.

Data augmentation has been applied with a specific degree of randomness: each technique had 50% of probability of being applied to a data sample. Random noise and pitch modification are uniformly distributed between 0.0 and 0.1 for each sample, while speed modification coefficient is uniformly distributed between 0.5 and 1.5 for each sample.


As result, the steps followed in the augmented feature pipeline are:
- Download the dataset;
- Remove not useful features such as accent, age, gender, etc, leaving just the audio array;
- Change the sample rate to 48000 to 16000;
- Apply Data Augmentation;
- Apply [Whisper Feature Extractor Tiny](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperFeatureExtractor);
- Apply [Whisper Tokenizer Tiny](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperTokenizer).

## Training Pipeline
The models have been trained following the [notebook](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Lab2/swedish_fine_tune_whisper.ipynb) provided by the professor:
- A pretrained model is loaded and then trained on the Italian dataset using the [Seq2Seq Trainer](https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/trainer#transformers.Seq2SeqTrainer) class;
- The model is saved on drive and uploaded as repository on HuggingFace;
- The model is checkpointed and evaluated at every 1000 steps.

## Daily Feature Pipeline for Synthetic Data
- Data has been generate by taking random values from each column of the original cleaned dataset.
- There is a 50/50 probability for the synthetic passenger to survive or not.
- Data generated is then added to Hopsworks' feature group, on which the model is trained daily.
- The script runs on Modal daily.

## Batch Inference Pipeline
- Every day the script runs on Modal and evaluates the last passenger added in the dataset.
- Results in the form of a confusion matrix and a table with the last 4 predictions are saved on Hopsworks in order to be accessed from the [Gradio Monitor application](https://huggingface.co/spaces/GIanlucaRub/Titanic-monitor).
