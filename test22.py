import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pyautogui
import win32gui
import time
import cv2
from PIL import Image
tf.data.Dataset
from sklearn.model_selection import train_test_split
#load the data 
#prprocessing the data
data= tf.keras.utils.image_dataset_from_directory('dataset_symbols - Copy',image_size=(28,28), color_mode='grayscale',shuffle= True)
data=data.map(lambda x , y :(x/255,y))
data_iterator= data.as_numpy_iterator()
data_batch= data_iterator.next()
print(len(data))
print(len(data_batch))
# for image, label in data:

#     print(label)
 

print(data_batch[1])
fig,ax= plt.subplots(ncols=25,figsize=(20,20))
for idx,img in enumerate(data_batch[0][:25]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(data_batch[1][idx])
plt.show()    
# 0-alpha, 1-beta, 2-phi,

# print(data_batch[0].max())
# print(data_batch[0].min())
# print(data_batch[1])
# print(data_batch[1].min())
# print(data_batch[1].max
# ())
# print(data_batch[0].shape)
# print(data_batch[1].shape)


# #data train
# # print(len(data))

# train_size=int(len(data)*.7)
# val_size= int(len(data)*.2)
# test_size=int(len(data)*.1)+1

# train=data.take(train_size)
# val=data.skip(train_size).take(val_size)
# test= data.skip(train_size+val_size).take(test_size)

# # #deep learnig model
# # #adding activation function and layers 

# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(32,(3,3), activation = 'relu',input_shape=(28,28,1)))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))

# model.add(tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(15, activation = 'softmax'))
# print(model.summary())

# #model compilling
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history=model.fit(train, epochs=10, batch_size=32, validation_data= val)

# print(model.evaluate(test))
# loss, accuracy = model.evaluate(test)
# print("loss:",loss)
# print("accuracy:",accuracy)
# plt.subplot(2,1,1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower right')

# plt.show()

# model.save('handwritten_model_real2.h5')

model= tf.keras.models.load_model('handwritten_model_real.h5')

#appopen and  save screenshot

def open_application_with_size(application_path, window_width, window_height):
    # Open the application using os.startfile()
    os.startfile(application_path)
    
    # Wait for the application window to open
    pyautogui.sleep(2)  # Adjust this delay as needed
    
    # Get the handle of the window
    hwnd = win32gui.GetForegroundWindow()
    
    # Resize the window
    win32gui.MoveWindow(hwnd,0,0,window_width,window_height,True)
    # print(window_width,window_height)

    # Specify the coordinates of the region to capture
    x1, y1 = 270 ,280  # Top-left corner
    x2, y2 = 550, 550 # Bottom-right corner

    # Take a screenshot of the specified region

    time.sleep(20)
    # def waitforkey(message="eneter any key"):
    #     input(message)
    # waitforkey()    
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

    # Save the screenshot to a file
    screenshot.save(r'C:\Users\vaisakh\neural\resize2\screenshot.png')

# Example usage
application_path = (r"C:\Users\vaisakh\AppData\Local\Microsoft\WindowsApps\mspaint.exe")
window_widt = 10
window_height = 10

open_application_with_size(application_path, window_widt, window_height)

#conver to satandard 

directory= r'C:\Users\vaisakh\neural\resize2'
directory1= r'C:\Users\vaisakh\neural\imd2'
number=1
for  filename in os.listdir(directory):
    print(filename)
    image= os.path.join(directory,filename)

    img= cv2.imread(image)
    img=cv2.resize(img,(28,28))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    newname= f'imd2{number}.png'
    new_path= os.path.join(directory1,newname)
    print(number)
    print(new_path)
    imgg=Image.open(image)
    imgg.save(new_path)   
    cv2.imwrite(new_path,img)


#inputting immage

image_number= 1
while os.path.isfile(f"imd2/imd2{image_number}.png"):
    try :
        img= cv2.imread(f"imd2/imd2{image_number}.png")[:,:,0]
        img1=img
        
        img= np.invert(np.array([img]))        
        
        
        
        prediction= model.predict(img)
        max= np.argmax(prediction)
        if max== 0:
            print("alpha")
            #print("the value :")

        elif max== 1:
            print("beta") 
            #print("the value :")
        elif max== 2:
            print("gamma")
            #print("the value :")
        elif  max== 3:
            print("infinity")  
            #print("the value :")     
        elif max== 4:
            print("limit")
            #print("the value :")
        elif max== 5:
            print("log")
            #print("the value :")
        elif max== 6:
            print("phi") 
            #print("the value :")

        elif max==7:
            print("theta")
            #print("the value :")
        else:
            print("sorry couldn't understand")    



        # print(f"the label is {np.argmax(prediction)}")
        # img1=cv2.resize(img1,(500,500))
        # cv2.imshow("image",img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    

        
    except :
        print("error")
    finally :
        image_number= image_number+1
