from google.colab import drive

drive.mount('/content/gdrive')
# !pip install autokeras

from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio as imread
import os
from PIL import Image
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow
import autokeras as ak
from sklearn.model_selection import train_test_split

# # Read images of camera frames and convert to numpy array
# def load_images(path):
#     images = []
#     for filename in os.listdir(path):
#         img = Image.open(path + filename)
#         img_resize = img.resize((256, 256))
#         img = img_resize.convert("L") # GREY CHANNEL ( SINGLE CHANNEL )
#         img_array = np.array(img)
# #       img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
#         images.append(img_array)
#
#     images1 = np.array(images)
#     return images1,images
#
# # read csv file (transforms of animation movment)
# y_df = pd.read_csv('/content/gdrive/MyDrive/BishopThesis/AnimationTransform.csv')
# y_df
#
# # read all transform values in all frames for one joint(mixamorig:LeftForeArm)
# y_df[y_df['JointName']=='mixamorig:LeftForeArm'][:209][['position.x','position.y','position.z','rotation.x','rotation.y','rotation.z']]
# # 	position.x	position.y	position.z	rotation.x	rotation.y	rotation.z
# # 0	-1.239424	2.883109	-8.507330	0.007603	356.8396	358.47150
# # 18	-1.239424	2.883109	-8.507330	0.007603	356.8396	358.47150
#
#
# # Function to create dataset for 18 classes
# def create_data(path_images, path_csv, class_needed):
#   All_images , list_images= load_images(path_images)
#   y_df = pd.read_csv(path_csv)
#   # read all transform values in all frames for all joint in class_needed (mixamorig:LeftForeArm)
#   Y= y_df[y_df['JointName']==class_needed][:209][['position.x','position.y','position.z','rotation.x','rotation.y','rotation.z']]
#   y = np.array(Y)
#   return All_images,y
#
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # y = scaler.fit_transform(y)
# # creating dataet
# All_images,y = create_data('/content/gdrive/MyDrive/freelancing/Razieh/CameraFrame/','/content/gdrive/MyDrive/freelancing/Razieh/AnimationTransform.csv','mixamorig:LeftForeArm')
#
#
# # split train and test data  and use 20% data for test and 80% for train
# # x_train is numpy(cameraFrame) and y_train is numpy(6 transform data)
# x_train, x_test, y_train, y_test = train_test_split( All_images, y, test_size=0.20, random_state=12)
#
# # Reshape the images to have the channel dimension for RESNET.-----------------:
# # ResNet follows VGG’s full 3*3 convolutional layer design. The residual block has
# # two 3*3 convolutional layers with the same number of output channels. Each
# # convolutional layer is followed by a batch normalization layer and a ReLU
# # activation function. Then, we skip these two convolution operations and add
# # the input directly before the final ReLU activation function. This kind of design
# # requires that the output of the two convolutional layers has to be of the same shape as the input,
# # so that they can be added together. If we want to change the number of channels, we need to introduce an additional
# # 1*1 convolutional layer to transform the input into the
# # desired shape for the addition operation.
# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))
#
# print(x_train.shape)  # (60000, 28, 28)
# print(y_train.shape)  # (60000,)
# print(y_train[:3])
# # (167, 256, 256, 1)
# # (167, 6)
# # [[ 3.023898e+00  2.233975e+00 -3.202632e+00  3.574737e+02  1.680123e+02
# #    8.328323e+01]
# #  [ 9.037256e-03  2.185420e+00 -3.121105e-01  3.299579e+02  1.967989e+02
# #    4.738947e+01]
# #  [ 2.068523e+00  2.224609e+00 -2.152886e+00  3.267445e+02  1.938723e+02
# #    5.244590e+01]]
#
# # Stop training when a monitored metric has stopped improving.----------------
# # Assuming the goal of a training is to minimize the loss.
# # With this, the metric to be monitored would be 'loss', and mode would be 'min'.
# # A model.fit() training loop will check at end of every epoch whether the loss
# # is no longer decreasing, considering the min_delta and patience if applicable.
# # Once it's found no longer decreasing, model.stop_training is marked True and the training
# # terminates.
# # monitor: Quantity to be monitored.
# # patience: Number of epochs with no improvement after which training will be stopped.
# callback = EarlyStopping(monitor='loss', patience=5)
#
# # search the best model by autokeras model---------------------
# # AutoKeras(ak) is an implementation of AutoML for deep learning that uses neural architecture search.
# # AutoML refers to techniques for automatically discovering the best-performing model for a given dataset.
# # When applied to neural networks, this involves both discovering the model architecture and the hyperparameters used to train the model, generally referred to as neural architecture search.
# # AutoKeras is an open-source library for performing AutoML for deep learning models. The search is performed using so-called Keras models via the TensorFlow tf.keras API.
# # It provides a simple and effective approach for automatically finding top-performing models for a wide range of predictive modeling tasks, including tabular or so-called structured classification and regression datasets.
# input_node = ak.ImageInput()
# output_node = ak.ImageBlock(
#     # Only search ResNet architectures.
#     block_type="xception",
#     # Normalize the dataset.,
#     normalize='False',
#     # Do not do data augmentation.
#     augment=False,
# )(input_node)
#
# # to use the CategoricalToNumerical- to convert to regression model
# output_node = ak.RegressionHead()(output_node)
#
# # Feed the image regressor with training data.
# reg = ak.AutoModel(
#     inputs=input_node, outputs=output_node, overwrite=True, max_trials=1 # hyper parmater tuning
# )
#
# # perform the search
# reg.fit(x_train, y_train, callbacks = [callback],validation_data=(x_test, y_test),epochs=100)
# classname = 'mixamorig:LeftForeArm'
#
# # export my model the best model found by AutoKeras as a Keras Model.
# model = reg.export_model()
#
# print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>
# modelname = '/content/gdrive/MyDrive/BishopThesis/models_trained/'+classname+'_model'
# modelname_save = '/content/gdrive/MyDrive/BishpopThesis/models_trained/'+modelname+'.h5'
# try:
#     model.save(modelname, save_format="tf")
# except Exception:
#     model.save(modelname_save)
#
# # Trial 1 Complete [00h 02m 59s]
# # val_loss: 20966.84765625
# #
# # Best val_loss So Far: 20966.84765625
# # Total elapsed time: 00h 02m 59s
# # INFO:tensorflow:Oracle triggered exit
# # Epoch 1/100
# # 6/6 [==============================] - 9s 874ms/step - loss: 11729.7109 - mean_squared_error: 11729.7109 - val_loss: 23704.3984 - val_mean_squared_error: 23704.3984
# # Epoch 2/100
# # 6/6 [==============================] - 4s 707ms/step - loss: 3025.6477 - mean_squared_error: 3025.6477 - val_loss: 23614.6621 - val_mean_squared_error: 23614.6621
# # Epoch 3/100
# # 6/6 [==============================] - 4s 711ms/step - loss: 2285.4580 - mean_squared_error: 2285.4580 - val_loss: 23525.5977 - val_mean_squared_error: 23525.5977
# # Epoch 4/100
#
# # -----predict---------
# # Predict with the best model for LeftForeArm positions and rotations.
# predicted_y = reg.predict(x_test)
# print(predicted_y)
#
# # Evaluate the best model with testing data.
# print(reg.evaluate(x_test, y_test))
#
# # ..... [-5.4540648e+00 -1.0591292e+01 -1.4793110e+01  1.6863797e+02
# #    1.6705531e+02  5.6967323e+01]
# #  [-5.4540648e+00 -1.0591292e+01 -1.4793110e+01  1.6863797e+02
# #    1.6705531e+02  5.6967323e+01]]
# # 2/2 [==============================] - 1s 74ms/step - loss: 2307.2783 - mean_squared_error: 2307.2783
# # [2307.2783203125, 2307.2783203125]


# ===========Currently we only predicted positionxyz and rotationxyz for LeftForeArm. We need to predict for other 17 features and save all the best models for all the classes=========
import tensorflow as tf


def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = Image.open(path + filename)
        img_resize = img.resize((256, 256))
        img = img_resize.convert("L")
        img_array = np.array(img)
        #       img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        images.append(img_array)

    images1 = np.array(images)
    return images1, images


def create_data(path_images, path_csv, class_needed):
    All_images, list_images = load_images(path_images)
    y_df = pd.read_csv(path_csv)
    Y = y_df[y_df['JointName'] == class_needed][:209][
        ['position.x', 'position.y', 'position.z', 'rotation.x', 'rotation.y', 'rotation.z']]
    y = np.array(Y)
    return All_images, y


def Final_model(classname):
    All_images, y = create_data('/content/gdrive/MyDrive/BishopThesis/CameraFrame/',
                                '/content/gdrive/MyDrive/BishopThesis/AnimationTransform.csv', classname)

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    x_train, x_test, y_train, y_test = train_test_split(All_images, y, test_size=0.20, random_state=12)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    callback = EarlyStopping(monitor='loss', patience=5)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        # Only search ResNet architectures.
        block_type="xception",
        # Normalize the dataset.,
        normalize='False',
        # Do not do data augmentation.
        augment=False,
    )(input_node)
    output_node = ak.RegressionHead()(output_node)
    # Feed the image regressor with training data.
    reg = ak.AutoModel(
        inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
    )
    reg.fit(x_train, y_train, callbacks=[callback, lr_scheduler], validation_data=(x_test, y_test), epochs=100)
    predicted_y = reg.predict(x_test)
    filename = 'predicted_LeftForeArm.csv'
    model = reg.export_model()

    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>
    modelname = '/content/gdrive/MyDrive/BishopThesis/models_trained/' + classname + '_model'
    modelname_save = '/content/gdrive/MyDrive/BishopThesis/models_trained/' + modelname + '.h5'
    try:
        model.save(modelname, save_format="tf")
    except Exception:
        model.save(modelname_save)

    return predicted_y, x_test

# do model on 18 joint-----------
predicted_LeftForeArm,x_test_LeftForeArm = Final_model('mixamorig:LeftForeArm')
predicted_LeftFoot,x_test_LeftFoot = Final_model('mixamorig:LeftFoot')
predicted_LeftHandMiddle1,x_test_LeftHandMiddle1 = Final_model('mixamorig:LeftHandMiddle1')
predicted_RightForeArm,x_test_RightForeArm = Final_model('mixamorig:RightForeArm')
predicted_RightShoulder,x_test_RightShoulder = Final_model('mixamorig:RightShoulder')
predicted_RightUpLeg,x_test_RightUpLeg = Final_model('mixamorig:RightUpLeg')
predicted_Hips,x_test_Hips = Final_model('mixamorig:Hips')
predicted_LeftLeg,x_test_LeftLeg = Final_model('mixamorig:LeftLeg')
predicted_RightHand,x_test_RightHand = Final_model('mixamorig:RightHand')
predicted_Neck,x_test_Neck = Final_model('mixamorig:Neck')
predicted_RightLeg,x_test_RightLeg = Final_model('mixamorig:RightLeg')
predicted_LeftUpLeg,x_test_LeftUpLeg = Final_model('mixamorig:LeftUpLeg')
predicted_LeftShoulder,x_test_LeftShoulder = Final_model('mixamorig:LeftShoulder')
predicted_LeftHand,x_test_LeftHand = Final_model('mixamorig:LeftHand')
predicted_RightArm,x_test_RightArm = Final_model('mixamorig:RightArm')
predicted_Spine,x_test_Spine = Final_model('mixamorig:Spine')
predicted_RightFoot,x_test_RightFoot = Final_model('mixamorig:RightFoot')
predicted_Head,x_test_Head = Final_model('mixamorig:Head')



# Load the saved model and get the predictions
from keras.models import load_model
# loaded_model = load_model("model_autokeras/content/gdrive/MyDrive/BishopThesis/models_trained/mixamorig:LeftForeArm_model", custom_objects=ak.CUSTOM_OBJECTS)
loaded_model = load_model("/content/gdrive/MyDrive/BishopThesis/models_trained/mixamorig:LeftForeArm_model", custom_objects=ak.CUSTOM_OBJECTS)
predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
# convert to csv
pd.DataFrame(predicted_y).to_csv("/content/gdrive/MyDrive/BishopThesis/out1.cvs")
print(predicted_y)

# //////////////////////////////
# The model is giving average results even with advanced deep learning networks because:
#
# Dataset is small and even with augumentation its is difficult to get appropiate data ( Need more data )
# The dataset has more surroundings than the actual object so cropping the image which contains only the human will increase the model performance
# This is a regression problem and performing multiregression model using VGG16 ,Inception..etc wont have better performance as they are mostly designed for classification
# To solve the above problem a seperate model for each class-->for 2 Positions or rotations should be done
# Currently, I developed a single model which takes 18 different classes and gives all 6 classes as output but due to the upper limiting constraints the accuracy is not good. Once the above issues are solved , It can be done easily using the current code as reference
#
# NOTE: Another easy way is to create a regression machine learning models like Decision Tree, Random Forest..etc


# Below is the code for predicting only 2 labels for each class so that you can get accurate results.
import tensorflow as tf


def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = Image.open(path + filename)
        img_resize = img.resize((256, 256))
        img = img_resize.convert("L")
        img_array = np.array(img)
        #       img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        images.append(img_array)

    images1 = np.array(images)
    return images1, images


def create_data(path_images, path_csv, class_needed, labels1, labels2):
    All_images, list_images = load_images(path_images)
    y_df = pd.read_csv(path_csv)
    Y = y_df[y_df['JointName'] == class_needed][:209][[labels1, labels2]]
    y = np.array(Y)
    return All_images, y


def Final_model(classname, labels1, labels2):
    All_images, y = create_data('/content/gdrive/MyDrive/freelancing/Razieh/CameraFrame/',
                                '/content/gdrive/MyDrive/freelancing/Razieh/AnimationTransform.csv', classname, labels1,
                                labels2)

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    x_train, x_test, y_train, y_test = train_test_split(All_images, y, test_size=0.20, random_state=12)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    callback = EarlyStopping(monitor='loss', patience=5)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        # Only search ResNet architectures.
        block_type="xception",
        # Normalize the dataset.,
        normalize='False',
        # Do not do data augmentation.
        augment=False,
    )(input_node)
    output_node = ak.RegressionHead()(output_node)
    # Feed the image regressor with training data.
    reg = ak.AutoModel(
        inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
    )
    reg.fit(x_train, y_train, callbacks=[callback, lr_scheduler], validation_data=(x_test, y_test), epochs=100)
    predicted_y = reg.predict(x_test)
    filename = 'predicted_LeftForeArm.csv'
    model = reg.export_model()

    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>
    modelname = '/content/gdrive/MyDrive/freelancing/Razieh/models_trained/' + classname + labels1 + labels2 + '_model'
    modelname_save = '/content/gdrive/MyDrive/freelancing/Razieh/models_trained/' + modelname + labels1 + labels2 + '.h5'
    try:
        model.save(modelname, save_format="tf")
    except Exception:
        model.save(modelname_save)

    return predicted_y, x_test




predicted_LeftForeArm,x_test_LeftForeArm = Final_model('mixamorig:LeftForeArm','position.x','position.y')
predicted_LeftFoot,x_test_LeftFoot = Final_model('mixamorig:LeftFoot','position.x','position.y')
predicted_LeftHandMiddle1,x_test_LeftHandMiddle1 = Final_model('mixamorig:LeftHandMiddle1','position.x','position.y')
predicted_RightForeArm,x_test_RightForeArm = Final_model('mixamorig:RightForeArm','position.x','position.y')
predicted_RightShoulder,x_test_RightShoulder = Final_model('mixamorig:RightShoulder','position.x','position.y')
predicted_RightUpLeg,x_test_RightUpLeg = Final_model('mixamorig:RightUpLeg','position.x','position.y')
predicted_Hips,x_test_Hips = Final_model('mixamorig:Hips','position.x','position.y')
predicted_LeftLeg,x_test_LeftLeg = Final_model('mixamorig:LeftLeg','position.x','position.y')
predicted_RightHand,x_test_RightHand = Final_model('mixamorig:RightHand','position.x','position.y')
predicted_Neck,x_test_Neck = Final_model('mixamorig:Neck','position.x','position.y')
predicted_RightLeg,x_test_RightLeg = Final_model('mixamorig:RightLeg','position.x','position.y')
predicted_LeftUpLeg,x_test_LeftUpLeg = Final_model('mixamorig:LeftUpLeg','position.x','position.y')
predicted_LeftShoulder,x_test_LeftShoulder = Final_model('mixamorig:LeftShoulder','position.x','position.y')
predicted_LeftHand,x_test_LeftHand = Final_model('mixamorig:LeftHand','position.x','position.y')
predicted_RightArm,x_test_RightArm = Final_model('mixamorig:RightArm','position.x','position.y')
predicted_Spine,x_test_Spine = Final_model('mixamorig:Spine','position.x','position.y')
predicted_RightFoot,x_test_RightFoot = Final_model('mixamorig:RightFoot','position.x','position.y')
predicted_Head,x_test_Head = Final_model('mixamorig:Head','position.x','position.y')




predicted_LeftForeArm,x_test_LeftForeArm = Final_model('mixamorig:LeftForeArm','position.z','rotation.x')
predicted_LeftFoot,x_test_LeftFoot = Final_model('mixamorig:LeftFoot','position.z','rotation.x')
predicted_LeftHandMiddle1,x_test_LeftHandMiddle1 = Final_model('mixamorig:LeftHandMiddle1','position.z','rotation.x')
predicted_RightForeArm,x_test_RightForeArm = Final_model('mixamorig:RightForeArm','position.z','rotation.x')
predicted_RightShoulder,x_test_RightShoulder = Final_model('mixamorig:RightShoulder','position.z','rotation.x')
predicted_RightUpLeg,x_test_RightUpLeg = Final_model('mixamorig:RightUpLeg','position.z','rotation.x')
predicted_Hips,x_test_Hips = Final_model('mixamorig:Hips','position.z','rotation.x')
predicted_LeftLeg,x_test_LeftLeg = Final_model('mixamorig:LeftLeg','position.z','rotation.x')
predicted_RightHand,x_test_RightHand = Final_model('mixamorig:RightHand','position.z','rotation.x')
predicted_Neck,x_test_Neck = Final_model('mixamorig:Neck','position.z','rotation.x')
predicted_RightLeg,x_test_RightLeg = Final_model('mixamorig:RightLeg','position.z','rotation.x')
predicted_LeftUpLeg,x_test_LeftUpLeg = Final_model('mixamorig:LeftUpLeg','position.z','rotation.x')
predicted_LeftShoulder,x_test_LeftShoulder = Final_model('mixamorig:LeftShoulder','position.z','rotation.x')
predicted_LeftHand,x_test_LeftHand = Final_model('mixamorig:LeftHand','position.z','rotation.x')
predicted_RightArm,x_test_RightArm = Final_model('mixamorig:RightArm','position.z','rotation.x')
predicted_Spine,x_test_Spine = Final_model('mixamorig:Spine','position.z','rotation.x')
predicted_RightFoot,x_test_RightFoot = Final_model('mixamorig:RightFoot','position.z','rotation.x')
predicted_Head,x_test_Head = Final_model('mixamorig:Head','position.z','rotation.x')



predicted_LeftForeArm,x_test_LeftForeArm = Final_model('mixamorig:LeftForeArm','rotation.y','rotation.z')
predicted_LeftFoot,x_test_LeftFoot = Final_model('mixamorig:LeftFoot','rotation.y','rotation.z')
predicted_LeftHandMiddle1,x_test_LeftHandMiddle1 = Final_model('mixamorig:LeftHandMiddle1','rotation.y','rotation.z')
predicted_RightForeArm,x_test_RightForeArm = Final_model('mixamorig:RightForeArm','rotation.y','rotation.z')
predicted_RightShoulder,x_test_RightShoulder = Final_model('mixamorig:RightShoulder','rotation.y','rotation.z')
predicted_RightUpLeg,x_test_RightUpLeg = Final_model('mixamorig:RightUpLeg','rotation.y','rotation.z')
predicted_Hips,x_test_Hips = Final_model('mixamorig:Hips','rotation.y','rotation.z')
predicted_LeftLeg,x_test_LeftLeg = Final_model('mixamorig:LeftLeg','rotation.y','rotation.z')
predicted_RightHand,x_test_RightHand = Final_model('mixamorig:RightHand','rotation.y','rotation.z')
predicted_Neck,x_test_Neck = Final_model('mixamorig:Neck','rotation.y','rotation.z')
predicted_RightLeg,x_test_RightLeg = Final_model('mixamorig:RightLeg','rotation.y','rotation.z')
predicted_LeftUpLeg,x_test_LeftUpLeg = Final_model('mixamorig:LeftUpLeg','rotation.y','rotation.z')
predicted_LeftShoulder,x_test_LeftShoulder = Final_model('mixamorig:LeftShoulder','rotation.y','rotation.z')
predicted_LeftHand,x_test_LeftHand = Final_model('mixamorig:LeftHand','rotation.y','rotation.z')
predicted_RightArm,x_test_RightArm = Final_model('mixamorig:RightArm','rotation.y','rotation.z')
predicted_Spine,x_test_Spine = Final_model('mixamorig:Spine','rotation.y','rotation.z')
predicted_RightFoot,x_test_RightFoot = Final_model('mixamorig:RightFoot','rotation.y','rotation.z')
predicted_Head,x_test_Head = Final_model('mixamorig:Head','rotation.y','rotation.z')
