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
OUTPUT:
![image](https://user-images.githubusercontent.com/97940064/173813524-b5019ad7-5079-4772-b171-1edb21823f66.png)

