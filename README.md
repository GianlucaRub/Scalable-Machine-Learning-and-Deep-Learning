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
## Gradio Application
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

## Hyperparemeter Tuning
Due to the previously stated limitations, the dataset used as benchmark for the hyperparameter tuning is the evaluation dataset previously descripted. If we had more time (and space) we would have performed k-fold cross validation to have more robust results.
Moreover, we chosed to focus mainly on the smaller version of the Whisper family, the tiny model because it has less parameters and therefore requires less time to train.
Starting from the whisper tiny model we:
- Changed the learning rate, going from 1e-4 to 1e-6, where the default is 1e-5;
- Changed the weight decay, going from 0.0 to 0.3, where the default is 0.0;
- Put the dropout of the attention layer, the decoder layer and the encoder layer to 0.1, where the default is 0.0;
- Changed the number of attention heads to 8, where the default is 6.

However, the attempts to change the network structure were often unsuccessful since the weights associated with the new structure where not pretrained. With high probability having more data (and more time) would have lead to different results.

## Scoreboard


| Model Version | Evaluation Wer | Description |
|:-------------:|:--------------:|:------------|
| Tiny 1        | 43.2959        |Plain Whisper Tiny model     |
| Tiny 2        | 43.3930        |Weight Decay set to 0.3    |
| Tiny 3        | 43.2335        |Weight Decay set to 0.1     |
| Tiny 4        | 41.3547        |Weight Decay set to 0.1 and Learning Rate set to 5e-5|
| Tiny 5        | 41.2715        |Weight Decay set to 0.1 and Learning Rate set to 1e-4|
| Tiny 6        | 46.2770        |Weight Decay set to 0.1, Learning Rate set to 1e-4,attention dropout, encoder dropout and decoder dropout have been set to 0.1, the number of decoder attention heads and encoder attention heads have been set to 8|
| Tiny 7        | 97.5666        |Weight Decay set to 0.1, Learning Rate set to 1e-6,attention dropout, encoder dropout and decoder dropout have been set to 0.1, the number of decoder attention heads and encoder attention heads have been set to 8|


