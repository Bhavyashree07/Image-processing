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
Fiename: b1.jpg
Format: JPEG
Size: (1300, 1036)
Mode: RGB
Width: 1300
Height: 1036
                 



