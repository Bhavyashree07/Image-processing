# Image-processing
# 1. Develop a program to display grayscae image using read and write operations<br>
import cv2<br>
img=cv2.imread('f2.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/181447933-fc022a65-6523-409f-8b00-b7397d856a4a.png)

******************************************************************************************************************

# 2. Develop a program to display the image using matplotlib<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('b1.jpg')<br>
plt.imshow(img)<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/173811925-dd732c9c-369b-4776-8c72-2d1bf0331b09.png)

*****************************************************************************************************************

# 3.Develop a program to perform linear transformation -Rotation<br>
from PIL import Image<br>
img=Image.open("b1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br> 

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/173813524-b5019ad7-5079-4772-b171-1edb21823f66.png)

****************************************************************************************************************

# 4.Develop a program to convert color strings to RGB color values<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>

**OUTPUT:**<br>
(255, 255, 0)<br>
(255, 0, 0)

*******************************************************************************************************************

# 5.Write a program to create image using colors from PI import Image<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/173815528-dc64c021-01d9-4cb7-8b9c-bac8456ee96f.png)

********************************************************************************************************************

# 6.Develop a program to visualize the image using various colorspaces<br>
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

******************************************************************************************************************


# 7.Write a program to display the image attributes from PIL import Image<br>
image=Image.open('b1.jpg')<br>
print("Fiename:",image.filename)<br>
print("Format:",image.format)<br>
print("Size:",image.size)<br>
print("Mode:",image.mode)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close();<br>

**OUTPUT:**<br>
Fiename: b1.jpg<br>
Format: JPEG<br>
Size: (1300, 1036)<br>
Mode: RGB<br>
Width: 1300<br>
Height: 1036<br>

************************************************************************************************************************

# 8.Convert the original image to Gray scale and then to Binary<br>
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

************************************************************************************************************************                 

# 9.Resize the original image<br>
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

*****************************************************************************************************************

# 1.Develop a program to read image using URL<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.teahub.io/photos/full/41-417562_goldfish-fish-facts-wallpapers-pictures-download-gold-fish.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**OUTPUT:**<br>
![image](https://user-images.githubusercontent.com/97940064/175004957-89831e2c-4554-4ecd-8a4b-568037d2240b.png)

# 2.Write a program to mask and blur the image<br>
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

***********************************************************************************************************************************************************************
NEW IMAGE<br>
![image](https://user-images.githubusercontent.com/97940064/183877009-f73923a3-b30b-492b-9241-7ca31528d206.png)<br>
![image](https://user-images.githubusercontent.com/97940064/183877197-6abcf7bf-b883-453d-8adf-316d8b28d52b.png)<br>
![image](https://user-images.githubusercontent.com/97940064/183877420-ea47b173-e4e9-4cb6-8b39-b50693040617.png)<br>




# 3.Write a program to perform arithmatic operations on images<br>
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



 

# 4.Develop the program to change the image to different color space<br>
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



# 5.Program to create an image using 2D array<br>
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



# 6.Bitwise Operation(with two diff images)<br>
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

# 7.Bitwise Operation(with two same images)<br>
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


# 8.Blurring<br>
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

************************************************************************************************************************

# 9.Image Enhancement<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('B2.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

output<br>
![image](https://user-images.githubusercontent.com/97940064/178713085-c5f7473e-4b89-4657-ba00-5726feedd6f4.png)
![image](https://user-images.githubusercontent.com/97940064/178713202-bdd1adc0-d1d0-4d3c-ace2-8811be468415.png)
![image](https://user-images.githubusercontent.com/97940064/178713432-614b501a-a774-48a8-8ddc-38b92f36b234.png)
![image](https://user-images.githubusercontent.com/97940064/178713616-622ea4f7-da5f-4a70-b5f7-2e3e33d0cabd.png)
![image](https://user-images.githubusercontent.com/97940064/178713698-12b744c3-cf75-43b3-8f8e-f26a91f5e8f3.png)



********************************************************************************************************************

# 10.Morphology<br>
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


# 11.Original and Grayscale Image<br>
import cv2<br>
OriginalImg=cv2.imread('img1.jpg')<br>
GrayImg=cv2.imread('img1.jpg',0)<br>
isSaved=cv2.imwrite('‪‪E:\flwr\img1.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The image is successfully saved.')<br>
    
**OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178717721-5dc19722-d070-4533-817f-aadccd3c7244.png)
![image](https://user-images.githubusercontent.com/97940064/178698908-1a77a7c8-5853-4837-ac6a-5bc625689619.png)
![image](https://user-images.githubusercontent.com/97940064/178699096-2abf0ccd-23a5-41f4-853f-c9fe5998fbe8.png)
![image](https://user-images.githubusercontent.com/97940064/178719932-38267dd2-610c-4196-a522-b685220ea5db.png)



# 12.Graylevel slicing with background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
    
***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178708617-987ba38f-d4f6-4749-8e03-831605be4083.png)




 # 13.Graylevel slicing without background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('cat.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>   

***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178709499-97bd1796-f623-44ee-a9e2-e11965c53134.png)


# 14.Analyse the image data using HISTOGRAM(numpy)<br>

import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('n3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('n3.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>

***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178964986-85efcf34-c0cc-4f05-bc39-741e746d735c.png)

*********************************************************************************************************************
skimage

from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('n3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
image = io.imread('n3.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>

***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178965106-ab58db9d-accc-4392-9d78-d795f5b28d7a.png)


from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = cv.imread('n3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
ax = plt.xlabel('Intensity Value')<br>
ax = plt.ylabel('Count') <br>
plt.show()<br>

***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178969193-064e73f6-469e-4134-a8fc-70d5092456dc.png)

*****************************************************************************************************************************************
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('n3.jpg')<br>
_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>

***OUTPUT***<br>
![image](https://user-images.githubusercontent.com/97940064/178965185-e5ab0418-1eab-4786-a672-92cd255e432d.png)

****************************************************************************************************************************************
from matplotlib import pyplot as plt<br>
import numpy as np<br>
fig,ax = plt.subplots(1,1)<br>
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])<br>
ax.hist(a, bins = [0,25,50,75,100])<br>
ax.set_title("histogram of result")<br>
ax.set_xticks([0,25,50,75,100])<br>
ax.set_xlabel('marks')<br>
ax.set_ylabel('no. of students')<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/178966391-420c8d62-a122-45e8-b9ab-1cc668d9da6c.png)



# 15.Program to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>

%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('btrfly1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179950252-e75f9250-6eba-45a6-9770-f83f7ff10fd3.png)


negative=255-pic<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179950631-61a8759f-0faa-49ef-8618-5915631217de.png)

%matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('btrfly1.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179951215-fa34b4f7-dcfb-4382-91d8-918f975945e7.png)

import imageio<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('btrfly1.jpg')<br>
gamma=2.2<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179951442-72d376bf-aa8a-4b99-9071-0274fcbef7df.png)



# 16.Program to perform basic image manipulation:<br>
a) Sharpness<br>
b) Flipping<br>
c) Cropping<br>

from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
my_image=Image.open('lion.jpg')<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
sharp.save('D:/lion_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179965825-8a9b07e7-851e-405a-8912-4c00ed0d4114.png)

import matplotlib.pyplot as plt<br>
img=Image.open('lion.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
flip.save('D:/lion_sharpen.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179965985-f2c9e196-427b-4e82-9378-a58d3fd1efbe.png)
![image](https://user-images.githubusercontent.com/97940064/181442818-0b07e16f-1349-44c8-a889-b715e1fbf37b.png)



from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im=Image.open('lion.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
width,height=im.size<br>
im1=im.crop((750,200,1600,800))<br>
#im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

**OUTPUT**<br>
![image](https://user-images.githubusercontent.com/97940064/179966792-4a7920d4-abff-46e3-8b52-8807f68f3f5e.png)\


from PIL import Image<br>
import numpy as np<br>
w, h =512,512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:256, 0:256] = [255, 0, 0] # red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
img.show()<br>

OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/180201907-3bd87968-3eef-44bb-9725-7aae35007223.png)



from PIL import Image<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
w, h = 512,512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [155, 75, 200] <br>
data[100:200,100:200] = [100,200,210] <br>
data[200:300,200:300 ] = [155, 75, 200] <br>
data[300:400,300:400] = [100,200,210] <br>
data[400:500, 400:500] = [155, 75, 200] <br>
#red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
#img.save('my.png')<br>
#img.show()<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/180202476-9cc8cd5c-137e-4dc1-a54a-4f130dd16ea2.png)


EDGE DETECTION:<br>
   import cv2<br>
# Read the original image<br>
img = cv2.imread('B1.jpg')<br>
# Display original image<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>
# Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
# Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>
# Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
# Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
# Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
# Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>


OUTPUT:
![image](https://user-images.githubusercontent.com/97940064/186404021-cc80461c-9516-43dd-a7a0-2fadf7e4e67d.png)<br>
![image](https://user-images.githubusercontent.com/97940064/186404164-00efaf77-4167-4b31-9440-c1a555281c56.png)<br>
![image](https://user-images.githubusercontent.com/97940064/186404249-829facd0-b227-4f0b-9326-64f1ddfda8a9.png)<br>
![image](https://user-images.githubusercontent.com/97940064/186404304-2b675250-84fe-401a-af44-df99f4b36d63.png)<br>
![image](https://user-images.githubusercontent.com/97940064/186404365-c857eb56-ca32-42e3-a21f-e0e96a2612c2.png)<br>


USING PILLOW FUNCTIONS<br>

from PIL import Image, ImageChops, ImageFilter<br>
from matplotlib import pyplot as plt<br>
#Create a PIL Image objects<br>
x = Image.open("x.png")<br>
o = Image.open("o.png")<br>
#Find out attributes of Image Objects<br>
print('size of the image: ', x.size, ' colour mode:', x.mode)<br>
print('size of the image: ', o.size, ' colour mode:', o.mode)<br>
#plot 2 images one besides the other<br>
plt.subplot(121), plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>
#multiply images<br>
merged = ImageChops.multiply(x,o)<br>
#adding 2 images<br>
add = ImageChops.add(x,o)<br>
#convert colour mode<br>
greyscale = merged.convert('L')<br>
greyscale<br>

OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187871854-67490187-07d7-4499-aec6-2e020d0d1a06.png)

![image](https://user-images.githubusercontent.com/97940064/186653501-d4e1a714-a124-400a-911d-96a34efff4fe.png)

#More Attributes<br>
image = merged<br>
print('image size: ', image.size,
     '\ncolor mode:', image.mode,
     '\nimage width :',image.width, '| also represented by:', image.size[0],<br>
'\nimage height:',image.height, '| also represented by: ', image.size[1],)<br>
![image](https://user-images.githubusercontent.com/97940064/187879920-fad9ce62-1b5b-4fa8-94a2-2203ad2c99bb.png)

#mapping the pixels of the image so we can use them as coordinates<br>
pixel = greyscale.load()<br>

#a nested Loop to parse through all the pixels in the image<br>
for row in range(greyscale.size[0]):<br>
 for column in range(greyscale.size[1]):<br>
    if pixel[row, column] != (255): <br>
     pixel[row, column] = (0)<br>
greyscale<br>
![image](https://user-images.githubusercontent.com/97940064/187880140-adcb83ec-c563-4b1e-8ec0-954e76d6d9b3.png)<br>

#1. invert image<br>
invert = ImageChops.invert(greyscale)<br>
#2. invert by subtraction<br>
bg = Image.new('L', (256, 256), color=(255)) #create a new image with a solid white background <br>
subt = ImageChops.subtract(bg, greyscale) #subtract image from background<br>
#3. rotate<br>
rotate = subt.rotate(45)<br>
rotate<br>
![image](https://user-images.githubusercontent.com/97940064/187880272-a0897a24-1941-4828-be63-6f6ffe9b2dda.png)<br>

#gaussian blur<br>
blur = greyscale.filter(ImageFilter.GaussianBlur(radius=1))<br>
#edge detection<br>
edge = blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
![image](https://user-images.githubusercontent.com/97940064/187880394-1e2914a4-6c43-40f4-a0be-4fcc75ce5995.png)<br>

#change edge colours<br><br>
edge = edge.convert('RGB')<br>
bg_red = Image.new('RGB', (256,256), color = (255,0,0))<br>
filled_edge = ImageChops.darker(bg_red, edge) <br>
filled_edge<br>
![image](https://user-images.githubusercontent.com/97940064/187880487-bb68263a-ecb5-437e-9eba-1e7b3b0dfb9b.png)<br>


IMAGE RESTORATION<br>
import numpy as np
import cv2
import matplotlib.pyplot as plt
#Open the image.
img = cv2.imread('cat_damaged.png')
plt.imshow(img)
plt.show()
#Load the mask.
mask = cv2.imread('cat_mask.png', 0)
plt.imshow(mask)
plt.show()
#Inpaint.
dst = cv2.inpaint (img, mask, 3, cv2.INPAINT_TELEA)
#write the output.
cv2.imwrite('dimage_inpainted.png', dst)
plt.imshow(dst)
plt.show()

OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187883101-1648d232-26ec-4d4f-af8d-31063b9ebe9e.png)

import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
plt.rcParams['figure.figsize'] = (10, 8)<br>

def show_image (image, title='Image', cmap_type='gray'):<br>
        plt.imshow(image, cmap=cmap_type)<br>
        plt.title(title)<br>
        plt.axis('off')<br>
def plot_comparison(img_original, img_filtered, img_title_filtered):<br>
    fig, (ax1, ax2)= plt.subplots (ncols=2, figsize=(10, 8), sharex=True, sharey=True)<br>
    ax1.imshow(img_original, cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered, cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
    
from skimage.restoration import inpaint <br>
from skimage.transform import resize <br>
from skimage import color<br>

image_with_logo= plt.imread('imglogo.png')<br>
#Initialize the mask<br>
mask= np.zeros(image_with_logo.shape[:-1])<br>
#Set the pixels where the Logo is to 1 <br>
mask [210:272, 360:425] = 1<br>
#Apply inpainting to remove the Logo<br>
image_logo_removed = inpaint.inpaint_biharmonic (image_with_logo,mask, multichannel=True)<br>
#Show the original and Logo removed images<br>
plot_comparison (image_with_logo, image_logo_removed, 'Image with logo removed')<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187883483-41c0d565-aeea-46e9-9b73-7d0a8487cc0a.png)<br>

from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>
#Add noise to the image<br>
noisy_image = random_noise (fruit_image)<br>
#Show th original and resulting image<br>
plot_comparison (fruit_image, noisy_image, 'Noisy image')<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187884110-71a8b196-bfc2-4c62-a82f-09d5c5d27bff.png)<br>

from skimage.restoration import denoise_tv_chambolle<br>
noisy_image = plt.imread('noisy.jpg')<br>
#Apply total variation filter denoising<br> 
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>
#Show the noisy and denoised image <br>
plot_comparison (noisy_image, denoised_image, 'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/97940064/187884948-1c1e0ade-edbc-4dc1-b12b-2fd36f0d8d42.png)<br>


from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>
#Apply bilateral filter denoising <br>
denoised_image = denoise_bilateral (landscape_image, multichannel=True)<br>
#Show original and resulting images<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187885233-e25f09df-3c7b-43ab-984e-8920bde2fa3b.png)<br>

from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
face_image=plt.imread('face.jpg')<br>

#obtain the segmentation with 400 regions<br>
segments = slic(face_image, n_segments=400)<br>

#Put segments on top of original image to compare<br>
segmented_image = label2rgb(segments, face_image,kind='avg')<br>

#Show the segmented image<br>
plot_comparison (face_image, segmented_image, 'Segmented image, 400 superpixels')<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187885347-49c10cc7-b31b-4c49-8c25-7b548389d947.png)<br>

def show_image_contour (image, contours):<br>
    plt.figure()<br>
    for n, contour in enumerate(contours):<br>
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)<br>
    plt.imshow(image, interpolation='nearest', cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
    
 from skimage import measure, data<br>
#Obtain the horse image <br>
horse_image = data.horse()<br>
#Find the contours with a constant Level value of 0.8 <br>
contours = measure.find_contours (horse_image, level=0.8)<br>
#Shows the image with contours found <br>
show_image_contour (horse_image, contours)<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187885493-09e2b787-3653-403d-9380-0093b188bcc5.png)<br>

from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>
image_dices = imread('diceimg.png')<br>
#Make the image grayscale<br>
image_dices = color.rgb2gray(image_dices)<br>
#obtain the optimal thresh value<br>
thresh = threshold_otsu(image_dices)<br>
#Apply thresholding <br>
binary = image_dices > thresh<br>
#Find contours at a constant value of 0.8<br>
contours = measure.find_contours (binary, level=0.8)<br>
#Show the image<br>
show_image_contour (image_dices, contours)<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187885600-0e78dd17-4e2f-428e-9f50-5665ad3f3925.png)<br>


#Create List with the shape of each contour<br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>
#Set 50 as the maximum size of the dots shape<br>
max_dots_shape = 50<br>
#Count dots in contours excluding bigger than dots size<br>
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]<br>
#Shows all contours found <br>
show_image_contour (binary, contours)<br>
#Print the dice's number<br>
print('Dices dots number: {}.'.format(len (dots_contours)))<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187885724-75fa6006-125d-4519-9e58-24f918d74a0c.png)<br>


Implement a program to perform various edge detection techniques<br>
a) Canny Edge detection<br>
#Canny Edge detection<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
plt.style.use('seaborn')<br>
loaded_image = cv2.imread("Image.png")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)<br>
edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>
plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image, cmap="gray")<br>
plt.title("Original Image")<br>
plt.axis("off")<br>
plt.subplot(1,3,2)<br>
plt.imshow(gray_image, cmap="gray")<br>
plt.axis("off")<br>
plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>
plt.title("Canny Edge Detected Image")<br>
plt.show()<br>
OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940064/187892757-a69fe5cc-25c3-440a-b3cb-52e142050039.png)

b) Edge detection schemes - the gradient (Sobel - first order derivatives)
based edge detector and the Laplacian (2nd order derivative, so it is
extremely sensitive to noise) based edge detector.<br>
#Laplacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Loading image
#img0 = cv2.imread('SanFrancisco.jpg',) 
img0= cv2.imread('Image.png',)
#converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# remove noise
img= cv2.GaussianBlur (gray, (3,3),0)
#convolute with proper kernels
laplacian= cv2.Laplacian (img, cv2.CV_64F) 
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # x 
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y
plt.subplot(2,2,1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray') 
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
OUTPUT
![image](https://user-images.githubusercontent.com/97940064/187892929-e65c9ce1-9c28-49a6-a9a5-1b63657f04a7.png)

c) Edge detection using Prewitt Operator<br>
#Edge detection using Prewitt operator
import cv2
import numpy as np
from matplotlib import pyplot as plt
img= cv2.imread('Image.png')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)
#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) 
img_prewittx= cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
cv2.imshow("Original Image", img)
cv2.imshow("Prewitt x", img_prewittx)
cv2.imshow("Prewitt y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/97940064/187896462-50e18889-b068-4064-8fad-1ea4f2947fc6.png)
![image](https://user-images.githubusercontent.com/97940064/187896521-d4cd5080-6567-4ed2-803e-303a15615196.png)
![image](https://user-images.githubusercontent.com/97940064/187896717-a0ff38d8-ce11-4fea-9652-51b2bfbbf37b.png)
![image](https://user-images.githubusercontent.com/97940064/187896769-87085efc-28b5-4db8-a096-d0321c554b35.png)


#Roberts Edge Detection- Roberts cross operator
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt 
roberts_cross_v = np.array([[1,0],
                            [0,-1]])
roberts_cross_h = np.array([[0, 1],
                            [-1,0]])
img = cv2.imread("Image.png",0).astype('float64')
img/=255.0
vertical = ndimage.convolve(img, roberts_cross_v ) 
horizontal = ndimage.convolve( img, roberts_cross_h)
edged_img = np.sqrt( np.square (horizontal) + np.square(vertical))
edged_img*=255
cv2.imwrite("output.jpg",edged_img)
cv2.imshow("OutputImage", edged_img)
cv2.waitKey()
cv2.destroyAllwindows()










   







