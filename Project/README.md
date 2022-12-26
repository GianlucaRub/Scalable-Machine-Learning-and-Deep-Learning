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

## Gradio Applications
- The [User Interface WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution) takes the model from Huggingface and allows the user to upload a picture and get its horizontal and vertical resolution doubled. It means that the number of pixels is increased by a factor of 4. Since the model has been trained on pictures with a fixed size (from 128x128 to 256x256), the input picture is transformed using a sliding window. However, this approach led the output picture to have some artifacts on the delimiting regions of each square. As result, we had to predict more squares and overlap some regions.
- The [Monitor WebApp](https://huggingface.co/spaces/GIanlucaRub/DoubleResolution-Monitor), takes the Nasa picture of the day and doubles its resolution. The picture is obtained by using their [API](https://github.com/nasa/apod-api). To update the page with the new picture, we added a button that calls the API to obtain the new picture. Sometimes the picture of the day is a video, in that case, we use the API to download the picture of a valid day.
- Since we are hosting both webapps on a free server, we do not have access to a GPU and we do not have many CPU cores, as result, using the model for pictures with high resolution may take several minutes.
## Feature Pipeline
The dataset used is [Google Open Image Dataset](https://storage.googleapis.com/openimages/web/download.html). 
It is made by around 1.9 millions pictures. 
The images are split into train (1,743,042), validation (41,620), and test (125,436) sets.  
The images are rescaled to have at most 1024 pixels on their longest side and at least 256 pixels on 
the shortest side, while preserving their original aspect-ratio. The total size is 561GB. The training set is dividend into 16 different parts.

Theoretically the feature pipeline would run before the training pipeline and would resize each picture 
to 256x256 while maintaining its proportions, however Colab has some problem with the synchronization of the 
VM with Google Drive when using many files. As result, the dataset has been downloaded and preprocessed 
in the training pipeline since it required less time. In fact, dowloading and storing one part of the dataset on 
Google Drive, requires more time than a night, while doing the same on the local hard disk of 
the notebook required just 20 minutes, which is acceptable.
## Training Pipeline
Since we had a very big dataset, it was not possible to train a model using it in its entirety. 
Training the model one part (1/16) of the dataset for one epoch required around 40 minutes, therefore training the 
model for one epoch on the entire dataset required more than 10 hours. 
In this situation it was not even possible to use checkpointing.
Inspired by the bagging technique, we trained 16 different models on each part of the dataset. The final output would be their averaged prediction.
Following this technique we did not use any regularization technique, since bagging requires complex models that overfit the data.
However, the error differenence between train and validation was generally low, meaning that the generalization capabilities of our models were good.
In this way we solved the problem of having too much data, and we were also able to parallelize the training. We trained two models at the same time.
The model structure is similar to a U-Net, but it has, obviously, the final layers that are up-sampled, resulting in a bigger size than the input.
We used skip connections to propagate better the gradient.
The framework used was tensorflow.
The loss function used was Binary Crossentropy, since it is well suited for the pixel representation. The optimizer used was Adam.

### Model Structure
![](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Project/Material/model_structure.png?raw=true)
The model is a fully convolutional neural network, it has 11,563,655 parameters.
The encoder part is made by convolutional layers and maxpooling layers. 
While the decoding part is made by convolutional transpose layers that have their input concatenated with the output of the
same size of the encoding part. At the end there are three convolutional layers.

## Ensemble Pipeline
The last step of our model development is to ensemble al the 16 trained models by averaging their predictions.
The resulting model was stored in huggingface and tested on the test set.
The test loss was 0.5021, however despite appearing high it is not the case. In fact, the binary crossentropy loss
is not zero when the real output value is not exactly 0 or 1, like it is for the pixels. Moreover, we computed the
loss on the same dataset by giving as input the same image of the output and not applying any transformation.
The resulting error was 0.4928, just 1.88% lower.
### Ensemble Model Structure
![](https://github.com/GianlucaRub/Scalable-Machine-Learning-and-Deep-Learning/blob/main/Project/Material/ensemble_model_structure.png?raw=true)
