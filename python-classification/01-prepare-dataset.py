import os
import random
import shutil

splitsize = .85
categories = []

source_folder = "Data-Sets"
folders = os.listdir(source_folder)
print(source_folder)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

# create target folder
target_folder = "Data-Sets/dataset_for_model"
existsDataSetPath = os.path.exists(target_folder)
if existsDataSetPath==False:
    os.mkdir(target_folder)

# split data for training and validation
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []
    
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is 0 length ......")
    print(len(files))

    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files)) 
    trainingSet = shuffleSet[0:trainingLength] # shuffling for all the images
    validSet = shuffleSet[trainingLength:]

    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)

    for filename in validSet:
        thisFile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination)

trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

# create target folders
existsDataSetPath = os.path.exists(trainPath)
if existsDataSetPath == False:
    os.mkdir(trainPath)

existsDataSetPath = os.path.exists(validatePath)
if existsDataSetPath == False:
    os.mkdir(validatePath)

# run through categories / folders of datasets
    for category in categories:
        trainDestPath = trainPath + "/" + category
        validateDestPath = validatePath + "/" + category

        if os.path.exists(trainDestPath)==False:
            os.mkdir(trainDestPath)
        if os.path.exists(validateDestPath)==False:
            os.mkdir(validateDestPath)

        sourcePath = source_folder + "/" + category + "/"
        trainDestPath = trainDestPath + "/"
        validateDestPath = validateDestPath + "/"

        print("Copy from : " + sourcePath + " to : " + trainDestPath + " and " + validateDestPath)

        split_data(sourcePath, trainDestPath, validateDestPath, splitsize)