import cv2
import os.path as f
import  numpy as np

def calc(im):
  #imname = "./images/out_"+"{}.jpg".format(i)
  #imsou = "./pot/{}.jpg".format(i)

  imname=im
  if(f.isfile(imname)):
     print("processing {}".format(imname))
     im=cv2.imread(imname,0)
    # ims = cv2.imread(imsou, 0)
     x=im.shape[0]
     y=im.shape[1]

     r = 11 / im.shape[1]
     returnValue=""
     dim = (11,int(im.shape[0] * r) )
     nwim=cv2.resize(im, dim, interpolation=cv2.INTER_NEAREST)
     img = np.zeros([x, y * 2, 3], dtype=np.uint8)
     cols = nwim.sum(axis=0,)
     cols=np.sum(nwim,where=nwim>90,axis=0)

     cv2.imwrite('images/out_r{}.jpg'.format(0), nwim)



     res = np.where(cols == np.amax(cols))

     left = False
     mid = False
     right =False
     regions=[]
     regions.append(cols[:5].sum())
     regions.append(cols[4:7].sum())
     regions.append(cols[7:].sum())
     reg_max=[]
     reg_max = np.where(regions == np.amax(regions))
     result=False
     print("image {} result: ".format(0))
     for e in range(0,2):
        if regions[e]==regions[e+1]:
           result=True
           break
        else:
           result=False
     print("reg:", regions)
     print('reg: ',result)
     bo=np.amax(reg_max)
     bo = np.amax(reg_max)
     con3=np.logical_or(reg_max[0] == 0 , res[0]<5)
     con2 = np.logical_or((reg_max[0] == 2), (res[0] > 6))
     con1= np.logical_and(reg_max[0] == 1 , res[0] >= 4 , res[0] < 7)
     print(con1)
     if (con1.any()):
        print("middle")
        returnValue = "Mnot"

     elif (con2.any()):
        i=[]
        i.append(cols[6:8].sum())
        i.append(cols[7:9].sum())
        i.append(cols[8:10].sum())
        print(i)
        i_max = np.where(i == np.amax(i))
        if(i_max==2):
           print(' too right')
           returnValue="Rnot"
        else:
           print('right ok')
           returnValue="Rok"
     elif(con3.any()):
        i = []
        i.append(cols[0:2].sum())
        i.append(cols[1:3].sum())
        i.append(cols[2:4].sum())
        print(i)
        i_max = np.where(i == np.amax(i))
        if (i_max == 0):
           returnValue = "Lnot"
           print(' too left')
        else:
           returnValue = "Lok"
           print('left ok')
     return returnValue










