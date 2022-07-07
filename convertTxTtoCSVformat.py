# create keras model for the neural network and train it to read the images and predict the transforms
# convert text file in C:\Users\18195\Desktop\AnimationTransform.txt to the csv file
# the csv file is used to train the model

# load the text file
import csv

# read the text file
with open('C:\\Users\\18195\\Desktop\AnimationTransform.txt', 'r') as f:
    reader = csv.reader(f,delimiter=' ')
    data = list(reader)
    # create the csv file and seprate each element  with space delimeter into different columns
    with open('C:\\Users\\18195\\Desktop\AnimationTransform.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


    # create the keras model
    # from keras.models import Sequential
    

    




