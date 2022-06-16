# Image-processing
1. Develop a program to display grayscae image using read and write operations<br>
import cv2<br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
c2.destroyAllwindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/173814001-1d10f995-d4dc-4dad-9dfc-1f82573af136.png)




2. Develop a program to display the image using matplotlib<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('b1.jpg')<br>
plt.imshow(img)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/173811925-dd732c9c-369b-4776-8c72-2d1bf0331b09.png)


3.Develop a program to perform linear transformation -Rotation<br>
from PIL import Image<br>
img=Image.open("b1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/173813524-b5019ad7-5079-4772-b171-1edb21823f66.png)



4.Develop a program to convert color strings to RGB color values<br>

from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
OUTPUT:<br>
(255, 255, 0)
(255, 0, 0)



5.Write a program to create image using colors from PI import Image<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/173815528-dc64c021-01d9-4cb7-8b9c-bac8456ee96f.png)



6.Develop a program to visualize the image using various colorspaces<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('b1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/173816743-61595013-6f2e-40fc-a47f-fdf4a14c6979.png)




7.Write a program to display the image attributes from PIL import Image<br>
image=Image.open('b1.jpg')<br>
print("Fiename:",image.filename)<br>
print("Format:",image.format)<br>
print("Size:",image.size)<br>
print("Mode:",image.mode)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close();<br>
OUTPUT:<br>
Fiename: b1.jpg<br>
Format: JPEG<br>
Size: (1300, 1036)<br>
Mode: RGB<br>
Width: 1300<br>
Height: 1036<br>



8.Convert the original image to Gray scale and then to Binary<br>
import cv2<br>
img=cv2.imread('f1.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>

img=cv2.imread('f1.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>

ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940064/174050027-2a29459c-8812-4729-a492-4d914420be4d.png)
![image](https://user-images.githubusercontent.com/97940064/174050216-523499ca-5b7a-4237-9b7d-c9ab4abe8905.png)
![image](https://user-images.githubusercontent.com/97940064/174050400-4d2b4ced-77f5-49d9-be4e-e10978764fb4.png)

                 

9.Resize the original image<br>
import cv2<br>
img=cv2.imread('p1.jpg')<br>
print('original image lenght width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
original image lenght width (800, 1280, 3)
![image](https://user-images.githubusercontent.com/97940064/174049340-a338dd0a-40bb-44d1-afd1-28f98fcb71dc.png)
![image](https://user-images.githubusercontent.com/97940064/174048924-3b7ad009-8a69-42a5-a755-acb81c830606.png)



