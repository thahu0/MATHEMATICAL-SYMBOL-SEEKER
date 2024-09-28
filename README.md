# MATHEMATICAL-SYMBOL-SEEKER
 a project that involves symbol recognition using a convolutional neural network (CNN) model in TensorFlow.
 The workflow includes loading the dataset, preprocessing images, building a CNN, training the model, saving/loading it, and then using PyAutoGUI to interact with an external application (like MS Paint) for capturing screenshots and processing images.
 Dataset Loading & Preprocessing

You are loading a dataset of symbols from a directory using image_dataset_from_directory, resizing images to 28x28, and converting them to grayscale. You then normalize the pixel values by dividing by 255 to scale them between 0 and 1.

python

data = tf.keras.utils.image_dataset_from_directory('dataset_symbols - Copy', image_size=(28,28), color_mode='grayscale', shuffle=True)
data = data.map(lambda x, y: (x/255, y))

CNN Model Definition

You define a CNN using tf.keras.Sequential, which includes three convolutional layers with ReLU activations, max-pooling layers, and two fully connected dense layers. The final layer is a softmax layer for multi-class classification.

python

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')  # Assuming 15 classes
])

Model Compilation and Training

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss, which is appropriate for multi-class classification problems. After training, you save the model.

python

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train, epochs=10, batch_size=32, validation_data=val)
model.save('handwritten_model_real2.h5')

Screenshot Capturing & Image Processing

You use PyAutoGUI and win32gui to capture screenshots from an application like MS Paint. The captured images are resized to 28x28, converted to grayscale, and saved for model prediction.

python

screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
screenshot.save(r'C:\Users\vaisakh\neural\resize2\screenshot.png')

Image Recognition Loop

You load images from the specified directory and use the trained CNN model to make predictions. Based on the model's output, it prints the symbol name.

python

img = cv2.imread(f"imd2/imd2{image_number}.png")[:,:,0]
img = np.invert(np.array([img]))
prediction = model.predict(img)
max = np.argmax(prediction)

Suggestions

    Model Evaluation: Consider visualizing the training and validation loss/accuracy to better understand the model’s performance over epochs.
    Data Augmentation: Adding data augmentation like rotations or flips might help improve model generalization.
    Screenshot Capturing: Adding dynamic delays or a way to confirm the application window is ready can ensure accurate screenshots.
    Error Handling: Consider enhancing error handling when processing images or making predictions to avoid silent failures.
