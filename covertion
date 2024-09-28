import cv2 
import os
from PIL import Image


directory= r"C:\Users\vaisakh\neural\imd1"
directory1= r"C:\Users\vaisakh\neural\resize2"
number=400
for  filename in os.listdir(directory):
    print(filename)
    image= os.path.join(directory,filename)

    img= cv2.imread(image)
    img=cv2.resize(img,(45,45))

    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    newname= f'imd2{number}.jpg'
    new_path= os.path.join(directory1,newname)
    print(number)
    print(new_path)
    #imgg=Image.open(image)
    #imgg.save(new_path)  
    cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
    #cv2.imwrite(new_path,img)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    number += 1
    print(number)


# img_num=1
# while os.path.isfile(f"imd1/imd{img_num}.jpg"):
#     try:
#         img= cv2.imread(f"imd1/imd{img_num}.jpg")
#         img=cv2.resize(img,(28,28))
#         img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
#         cv2.imwrite(f"imd1/imd{img_num}.jpg",img)
#         cv2.imshow('image',img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         img_num=img_num+1
#     finally:
#         print("error")    
    
