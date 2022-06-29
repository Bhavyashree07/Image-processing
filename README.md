# Image-processing
1. Develop a program to display grayscae image using read and write operations<br>
import cv2<br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
c2.destroyAllwindows()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/173814001-1d10f995-d4dc-4dad-9dfc-1f82573af136.png)




2. Develop a program to display the image using matplotlib<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('b1.jpg')<br>
plt.imshow(img)<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/173811925-dd732c9c-369b-4776-8c72-2d1bf0331b09.png)


3.Develop a program to perform linear transformation -Rotation<br>
from PIL import Image<br>
img=Image.open("b1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT:**<br>
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

**OUTPUT:**<br>
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

**OUTPUT:**<br>
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

**OUTPUT:**<br>
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

**OUTPUT:**<br>
original image lenght width (800, 1280, 3)
![image](https://user-images.githubusercontent.com/97940064/174049340-a338dd0a-40bb-44d1-afd1-28f98fcb71dc.png)
![image](https://user-images.githubusercontent.com/97940064/174048924-3b7ad009-8a69-42a5-a755-acb81c830606.png)


**1.Develop a program to read image using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.teahub.io/photos/full/41-417562_goldfish-fish-facts-wallpapers-pictures-download-gold-fish.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175004957-89831e2c-4554-4ecd-8a4b-568037d2240b.png)

**2.Write a program to mask and blur the image**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175014859-dc70d075-50ac-44b2-9bcc-0c54881d2d26.png)

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(8,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>


**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175015103-25dd2b09-bae3-402d-b3ad-e482a88c08c3.png)

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175015412-cf4a1c44-070f-424e-b312-a2364619524c.png)

final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175015768-71f7bcac-3d95-479c-a506-2db2778c358b.png)

blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175015972-6ddafcaa-562d-4ecd-98fb-b97054aed6f9.png)




**Write a program to perform arithmatic operations on images**
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>

img1=cv2.imread('f1.jpg')<br>
img2=cv2.imread('f2.jpg')<br>

fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg4)<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/175022506-d80eea4d-f08d-4ec4-8984-a006e2399e4f.png)
![image](https://user-images.githubusercontent.com/97940064/175282993-74e4f576-56ae-4361-b22d-3fabc976e5b7.png)
![image](https://user-images.githubusercontent.com/97940064/175285829-69a2906d-5761-4911-b630-ed986725b6e5.png)



 

**4.Develop the program to change the image to different color space**<br>
import cv2 <br>
img=cv2.imread('E:\\b3.jpg')<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OOUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/175274059-e993fc1e-7a0d-469d-b8de-fcfc994730a2.png)
![image](https://user-images.githubusercontent.com/97940064/175274255-8ca880c7-708f-4cd3-84fa-0f3cb182cf8f.png)
![image](https://user-images.githubusercontent.com/97940064/175274845-9a941493-9fa5-4bbc-8d2d-8e586d466cef.png)
![image](https://user-images.githubusercontent.com/97940064/175275003-5479e091-7ffb-49e5-b5c1-6e86ddf5ec3d.png)
![image](https://user-images.githubusercontent.com/97940064/175275132-aeb6c615-bc67-412c-8e2a-fa58953c544b.png)



**5.Program to create an image using 2D array**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100] = [250,130,0]<br>
array[:,100:] = [0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/175282492-cb40387b-39bd-4453-9a41-1256d02d3fe7.png)



**Bitwise Operation(with two diff images)**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('f3.jpg')<br>
image2=cv2.imread('f4.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/176404531-a617d1c6-abbe-4f3d-a639-16620754c1f2.png)

**Bitwise Operation(with two same images)**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('f3.jpg')<br>
image2=cv2.imread('f3.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/176406214-981e8df6-3f65-4c02-8aed-a04b1ef8506f.png)


**Blurring**<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('B2.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
median=cv2.medianBlur(image,15)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/176421155-23df0a8f-4a6f-460b-98d0-9f77ca99e004.png)
![image](https://user-images.githubusercontent.com/97940064/176421392-eae5a4c8-6d65-40cb-95db-497626d58605.png)
![image](https://user-images.githubusercontent.com/97940064/176421613-6759ca08-0829-470a-b340-28deda9c3455.png)
![image](https://user-images.githubusercontent.com/97940064/176421728-cdfb9c94-5b9f-4fe0-826e-9e16e95cb3d8.png)

























***Morphology***<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('B2.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

**OUTPUT**
![image](https://user-images.githubusercontent.com/97940064/176426010-f94f156c-fb2a-495e-9067-a51bceb3a5b0.png)
