# ID2223 Scalable Machine Learning and Deep Learning
## Project: Doubling pictures' resolution
**Professor:**
Jim Dowling

**Students:**

- Gianluca Ruberto
- Fabio Camerota

## Task description
- Choose a [real time data source](https://apod.nasa.gov/apod/ap221208.html) that is update regularly;
- Define the prediction problem that you will solve based on data from the [data source](https://storage.googleapis.com/openimages/web/download.html);
- Define a data model with the features and label(s) of your data;
- Build a feature pipeline to create the features/label(s) from the raw data;
- Build a training pipeline to train a model using your features/label(s);
- Build an [inference pipeline](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution-Monitor) to make predictions using new data and build a [User Interface](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution) to communicate the results to a user.
## Infrastructure Model
- Feature Pipeline. It runs on [Colab](https://colab.research.google.com/) and stores there the output.
- Training Pipeline, which is the same for all models, but changes the part of dataset used. It runs on Colab and stores there the output.
- Ensemble Pipeline, it creates the final model by averaging the prediction of the previously trained models. The model is tested and then stored on [Huggingface](https://huggingface.co/GIanlucaRub/doubleResFinal)
- Gradio Applications:
  - [User Interface WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution), it allows the user to upload a picture and get its resolution augmented. It is hosted on HuggingFace.
  - [Monitor WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution-Monitor), it doubles the resolution of the Nasa picture of the day. It is hosted on Huggingface.

## Gradio Application
- The [User Interface WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution) takes the model from Huggingface and allows the user to upload a picture and get its horizontal and vertical resolution doubled. It means that the number of pixels is increased by a factor of 4. Since the model has been trained on pictures with a fixed size (from 128x128 to 256x256), the input picture is transformed using a sliding window. However, this approach led the output picture to have some artifacts on the delimiting regions of each square. As result, we had to predict more squares and overlap some regions.
- The [Monitor WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution-Monitor), takes the Nasa picture of the day and doubles its resolution. The picture is obtained by using their [API](https://github.com/nasa/apod-api). To update the page with the new picture, we added a button that calls the API to obtain the new picture. Sometimes the picture of the day is a video, in that case, we use the API to download the picture of a valid day.
- Since we are hosting both webapps on a free server, we do not have access to a GPU and we do not have many CPU cores, as result, using the model for pictures with high resolution may take several minutes.
## Feature Pipeline
The dataset used is [Google Open Image Dataset](https://storage.googleapis.com/openimages/web/download.html). It is made by around 1.9 millions pictures. 
The images are split into train (1,743,042), validation (41,620), and test (125,436) sets.  
The images are rescaled to have at most 1024 pixels on their longest side and at least 256 pixels on 
the shortest side, while preserving their original aspect-ratio. The total size is 561GB. The training set is dividend into 16 different parts.

Theoretically the feature pipeline would run before the training pipeline and would resize each picture 
to 256x256 while mantaining its proportions, however Colab has some problem with the synchronization of the 
VM with Google Drive when using many files. As result, the dataset has been downloaded and preprocessed 
in the training pipeline since it required less time. In fact, dowloading and storing one part of the dataset on 
Google Drive, requires more time than a night, while doing the same on the local hard disk of the notebook required just 20 minutes, which is acceptable.
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

However, training with the augmented dataset did not improve the results, probably because we were using a small part of the dataset and it resulted in overfitting. If we had more time (and space) we would have applied data augmentation to the entire dataset.
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
If nothing else is specified in the description, the starting model is the Whisper Tiny model and the dataset used for training is the first 10% of the total dataset. Scores in bold mean that there has been an improvement. The metric used for comparison is the Word Error Rate (WER), computed on the first 10% of the entire test dataset.

| Model Version | Evaluation WER | Description |
|:-------------:|:--------------:|:------------|
| [Tiny 1](https://huggingface.co/GIanlucaRub/whisper-tiny-it-1)        | 43.2959        |Plain Whisper Tiny model     |
| [Tiny 2](https://huggingface.co/GIanlucaRub/whisper-tiny-it-2)        | 43.3930        |Weight Decay set to 0.3    |
| [Tiny 3](https://huggingface.co/GIanlucaRub/whisper-tiny-it-3)        | **43.2335**    |Weight Decay set to 0.1     |
| [Tiny 4](https://huggingface.co/GIanlucaRub/whisper-tiny-it-4)        | **41.3547**    |Weight Decay set to 0.1 and Learning Rate set to 5e-5|
| [Tiny 5](https://huggingface.co/GIanlucaRub/whisper-tiny-it-5)        | **41.2715**    |Weight Decay set to 0.1 and Learning Rate set to 1e-4|
| [Tiny 6](https://huggingface.co/GIanlucaRub/whisper-tiny-it-6)        | 46.2770        |Weight Decay set to 0.1, Learning Rate set to 1e-4,attention dropout, encoder dropout and decoder dropout have been set to 0.1, the number of decoder attention heads and encoder attention heads have been set to 8|
| [Tiny 7](https://huggingface.co/GIanlucaRub/whisper-tiny-it-7)        | 97.5666        |Weight Decay set to 0.1, Learning Rate set to 1e-6,attention dropout, encoder dropout and decoder dropout have been set to 0.1, the number of decoder attention heads and encoder attention heads have been set to 8|
| [Tiny 8](https://huggingface.co/FCameCode/whisper-tiny-it-8)        | 56.9052        |Weight Decay set to 0.1, Learning Rate set to 1e-5,attention dropout, encoder dropout and decoder dropout have been set to 0.1, the number of decoder attention heads and encoder attention heads have been set to 8|
| [Tiny 9](https://huggingface.co/GIanlucaRub/whisper-tiny-it-9)        | 45.3272        |Weight Decay set to 0.1 and Learning Rate set to 1e-4, trained with data augmentation|
| [Tiny 10](https://huggingface.co/GIanlucaRub/whisper-tiny-it-10)       | 46.8178        |Trained with data augmentation|
| [Tiny 11](https://huggingface.co/FCameCode/whisper-tiny-it-11)       | 42.2768        |Trained on 25% of the entire dataset|
| [Small](https://huggingface.co/GIanlucaRub/whisper-small-it-3)         | **22.1090**    |Plain Whisper Small model|
## Final Test
The best performing model, Whisper Small, has been tested on the not previously used part of the entire test set (90% of the entire test set). In this way, it is possible to have a more reliable estimate of the performance of the model. Computing the evaluation on the final test set required more than 3 hours.
| Model Version | Final Test WER | Description |
|:-------------:|:--------------:|:------------|
| [Small](https://huggingface.co/GIanlucaRub/whisper-small-it-3)         | 16.2960    |Plain Whisper Small model|
## Conclusion
The result of the project are heavily biased from the resources available on Colab. In fact, for the first days, we were not able to preprocess the dataset, since Colab was not giving us resources. In the ideal case, we would have used the entire train dataset to perform cross validation on the different models. In this case we would have been able to test also the larger Whisper transformers and to perform a better hyperparameter tuning process, possibily creating custom architectures that had some weights not pretrained. Moreover during the data augmentation space we would have tried different combinations of noise.
However, with the resources available, we were able to improve the performance of the tiny model of 5%. Nevertheless, using a bigger model (240M parameters vs 37M parameters) resulted in better performance.
