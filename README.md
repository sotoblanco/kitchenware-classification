# Kitchenware Classification project

## Data preparation

This project started by:

1. Analyzing the data from the competition with some basic exploratory analysis. In this part of the project the distribution of the categories was evaluated, we didn't find any major imbalance between the six classes.

2. The data was splitted into train, validation and test set, since we want to make sure that the model is not overfitting. We perform adverserial validation of the validation set to make sure that the validation set is not too different from the test set. This helps to test our data accurately, in other words the results we get in the public leaderboard are not too different from the results we get in the private leaderboard.

This notebook was use to implement the adverserial validation. The results were that the validation set is not too different from the test set. 

[![Kaggle](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/code/pastorsoto/adversarial-validation-for-overfitting-detection/notebook)

![image](https://user-images.githubusercontent.com/46135649/218275717-e2982e12-9d37-46f1-b510-f307624caf45.png)

![image](https://user-images.githubusercontent.com/46135649/218275740-2db6349f-f479-4d20-98ec-06dc82d626fc.png)

![image](https://user-images.githubusercontent.com/46135649/218275751-fee825ef-bc52-4235-9656-84d9a0de4961.png)

The images shows that training, validation and testing datasets has a similar distribution of the classes. After that we test using adverserial validation technique to see if the model is able to distinguish between the training and validation set. The results are shown in the notebook.

## Model

To test different models and run several experiments we use Weights & Biases for experimental tracking.

A detail explanation of the set-up for running the experiment was made on this notebook. 

[![Kaggle](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/code/pastorsoto/experimental-tracking-with-w-b/notebook)


The architecture of the model is shown in the following image:

![image](https://user-images.githubusercontent.com/46135649/218276197-91ea5d8d-8e40-4551-a5da-c35ae761e0e3.png)

The model use Xception as a base model, and we add a global average pooling layer and a dense layer with 50 units and relu activation function. The regularization layer was added to avoid overfitting using 0.2 dropout rate.

The output layer has 6 units and linear activation function. The linear activation function was use to speed up the training process, later the predictions were transformed to probabilities using the softmax function using the following code: 

```python
logits_test = model.predict(test_generator)
tf.nn.softmax(logits_test).numpy()
```

The model was trained for 100 epochs with a batch size of 32. The model was trained using the Adam optimizer with a learning rate of 0.001. The model was trained using the categorical crossentropy loss function. The input size of the images was 550x550x3. Early stopping was used to avoid overfitting using patience of 5 epochs monitoring the validation loss.

## Structure of the repository

``app.py``: Python file that runs a streamlit app for the prediction service

``cloud_test.py``: Python file that runs the prediction service from AWS.

``convert_model.py``: Python file that converts the .h5 file into .tflite
    
``Dockerfile``: For deployment of the model in AWS as lambda function.

``kitchenware_v4_09_0.967.h5``: Model with TensorFlow (can be directly downloaded from kaggle)

``kitchenwaren-class.tflite``: Model with TensorFlow lite

``local_test.py``: Python file that runs the prediction service from the docker file locally.

``Notebook.ipynb``: Notebook for exploratory data analysis, creating and exporting the model.

``Pipfile`` and ``Pipfile.loc``: contains the dependencies to run the project.

``process_data.py``: Python script to process an url with the image and return a prediction

``test.py``: Python script to test the prediction service using AWS.

``train.py``: Python script to train the model and export it as a .h5 file.

## How to run 

- Clone the repo
- Download the data from kaggle
- Install the dependencies

```
pipenv install
```

-   Activate the virtual enviroment

```
pipenv shell
```

### Building the prediction model and service

Run the  `train.py`  file to obtain the best model for the training parameters as a  `.h5`  file and convert to tflite file.

## Enviroment set-up
pipenv

Install pipenv in your machine:

``pip install pipenv``

set the python version that you want to use:

``pipenv install --python 3.9``

install the libraries that you want to use:

```
pipenv install pandas tensorflow numpy matplotlib
```

This would create a ``Pipfile`` and a ``Pipfile.lock`` 

## Containerization

Run the docker file:

First build the model:

```
docker build -t kitchen-class-model .
```

Run the docker image

```
docker run -it --rm -p 8080:8080 kitchen-class-model:latest
```

Run the prediction service: Open a new command line (make sure you are running the docker file)

Activate the virtual enviroment

```
pipenv shell
python test.py
```

The `local_test.py` already have an image link to return a prediction (feel free to add the URL you want to test)

## Deployment

### Cloud deployment

AWS

**pre-requisets**  needs to have AWS CLI installed which is command line to interact with AWS ( I have a windows and working with WSL, so I download the cli using the linux command)

#### Elastic Container Registry:

Place to store your container

Create repo View push command

Go to security credentials and find the access key to configure your AWS

run in your command line:  `aws configure`  and type your credentials from the above step

run:

Create the repo to store the image

```
aws ecr create-repository --repository-name kitchen-class-images
```

Obtain the repositoryUri which look something like this:

```
9xxxxxxx23.dkr.ecr.us-west-2.amazonaws.com/kitchen-class-images
```

Set at the command line

```
$(aws ecr get-login --no-include)

ACCOUNT=9xxxxxxx2

REGION=us-west-2

REGISTRY=kitchen-class-images

PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=kitchen-class-model-v1-001

REMOTE_URI=${PREFIX}:${TAG}
```

Tag the image to AWS

```
docker tag kitchen-class-model:latest ${REMOTE_URI}
```
Push the docker image
```
docker push ${REMOTE_URI}
```

#### Lambda function

[![image](https://user-images.githubusercontent.com/46135649/207652437-dfd995f8-6135-4229-b6a2-38183d273afa.png)](https://user-images.githubusercontent.com/46135649/207652437-dfd995f8-6135-4229-b6a2-38183d273afa.png)

Browse the image

![image](https://user-images.githubusercontent.com/46135649/218279932-622916eb-97f7-4c82-a412-f83da49cb999.png)


For deep learning task we need to increase the time of the response and the memory allocated to perform the function.

We need to go configuration -> General configuration and change the timeout to 30 seconds and the memory to 1024

Test the service

![image](https://user-images.githubusercontent.com/46135649/218280982-7cfa705d-dc70-4767-aead-ab50d0f8d2ca.png)


#### API Gateaway

Create API

![image](https://user-images.githubusercontent.com/46135649/212447794-f9564c5b-002a-4053-bf6f-475e587fd04c.png)

Build REST API

![image](https://user-images.githubusercontent.com/46135649/212447831-94bd3a37-ca10-42c1-baf4-4730b57cd934.png)

Choose the protocol -> Create API

![image](https://user-images.githubusercontent.com/46135649/218280541-02fa3bd4-0b37-440b-952d-164727e8c536.png)


Create method -> select ``POST``
Integration type: Lambda
Select the **Lambda Function**

![image](https://user-images.githubusercontent.com/46135649/218280627-3f1bcdc2-c668-4b80-9b4f-3055cf9b93cb.png)


#### Deploy endpoint

Go to actions and click on **Deploy API**

[![image](https://user-images.githubusercontent.com/46135649/207659795-fddbf3a3-1dc3-4ca8-9680-02fa8b5a3574.png)](https://user-images.githubusercontent.com/46135649/207659795-fddbf3a3-1dc3-4ca8-9680-02fa8b5a3574.png)


Select the **stage name**

[![image](https://user-images.githubusercontent.com/46135649/207660014-9baef1b4-fdb6-4637-a044-0fad8a86e8d3.png)](https://user-images.githubusercontent.com/46135649/207660014-9baef1b4-fdb6-4637-a044-0fad8a86e8d3.png)


Now we just need to obtain the URL if you select a name for the POST you need to added at the end, if not you can use the url provided:  

[![image](https://user-images.githubusercontent.com/46135649/207660282-f9c17a53-aa2b-4c04-8c17-74efcb1b88ba.png)](https://user-images.githubusercontent.com/46135649/207660282-f9c17a53-aa2b-4c04-8c17-74efcb1b88ba.png)


## Demo

![image](https://user-images.githubusercontent.com/46135649/218281850-8a5f7df1-ca0f-48ee-89e2-eebb5ebb1f52.png)

To run the streamlit app just need to run in your terminal:

```
streamlit run app.py
```
