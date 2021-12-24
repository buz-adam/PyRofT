import cv2
import numpy as np

def cnt(x,q,z):
 im=cv2.imread(x)

 img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

  # visualize the binary image

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
 contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


 image_copy = im.copy()

 conts=[]
#lens equal 9313  4656
 for i in contours:

   conts.append(cv2.contourArea(i))

 j=conts.index(max(conts))
 cnts=[]
 for p in range(0,len(conts)):

   if len(str(int(conts[j])))==len(str(int(conts[p]))):
        cnts.append(p)
   elif conts[j]/10>800 and len(str(int(conts[j]/10)))==len(str(int(conts[p]))):
        cnts.append(p)
 print(conts)
 print(cnts)
 opacity=0.4
 cop=q.copy()
 for i in cnts:
  cv2.ellipse(z,cv2.fitEllipse(contours[i]),(0,255,0),-1)
  cv2.ellipse(cop, cv2.fitEllipse(contours[i]), (0, 255, 0), -1)
  cv2.addWeighted(cop, opacity, q, 1 - opacity, 0, q)

 #cv2.imshow('shapes',q)
 #cv2.imshow('z',z)
 cv2.imwrite('results/z.jpg',z)
 cv2.imwrite('results/q.jpg', q)
 #cv2.waitKey(0)
 return len(cnts)