from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

trainPath = "Data-Sets/dataset_for_model/train"
validPath = "Data-Sets/dataset_for_model/validate"

trainGenerator = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=(0,0.2)).flow_from_directory(validPath, target_size=(320, 320), batch_size=32)

validGenerator = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=(0,0.2)).flow_from_directory(trainPath, target_size=(320, 320), batch_size=32)

baseModel = MobileNetV3Large(weights="imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

predictionLayer = Dense(3, activation='softmax')(x) # number is for the number of classes

model = Model(inputs=baseModel.input, outputs=predictionLayer)

print(model.summary())

# freeze the layers of MobileNet
# model already trained, freeze all layers
for layer in model.layers[:-5]:
    layer.trainable = False

# Compile
optimizer = Adam(learning_rate = 0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

model.fit(trainGenerator, validation_data=validGenerator, epochs=5)
modelSavedPath = "Data-Sets/dataset_for_model/MoonV3.h5"
model.save(modelSavedPath)

# def main():
#     print("Hello from main!")

# if __name__ == "__main__":
#     main()