import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
import random
import math
import numpy as np
from collections import Counter 
class preprocess(object):

    def __init__(self, dataset_location=r" C:\Users\Madke\Pictures\New folder", batch_size=1,shuffle=False):
        self.location = dataset_location
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.seed = 28
        self.dirlist = os.listdir(dataset_location)
        self.length = len(self.dirlist)
        self.idx=os.listdir(dataset_location)
        # First argument specifies the location of the dataset,
    # batch_size indicates the number of input images on which operation has to be performed(It takes in integer as input),
    # shuffle can take boolean True/ False values. If True it picks up random indices of the input images every time and if false it will pick up indexes sequentially
    def _getitem_(self):
        img_list = listdir(self.location)
        img_list = sorted(img_list,key = len)
        a = random.sample(img_list,min(len(img_list),self.batch_size))
        ret_dict = {}
        if self.shuffle==True:
            for i in a:
                ret_dict[i] = plt.imread('{}/{}'.format(self.location,i))
        else:
            for i in range(min(len(img_list),self.batch_size)):
                ret_dict[img_list[i]] = plt.imread('{}/{}'.format(self.location,img_list[i]))
        return ret_dict
    def rescale(self,s):
        k = self._getitem_()    
        def bilinear_resize(image, height, width):      
            img_height, img_width = image.shape[:2]
            resized = np.empty([height, width])
            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
            for i in range(height):
                for j in range(width):
                    x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                    x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
                    x_weight = (x_ratio * j) - x_l
                    y_weight = (y_ratio * i) - y_l
                    a = image[y_l, x_l]
                    b = image[y_l, x_h]
                    c = image[y_h, x_l]
                    d = image[y_h, x_h]
                    pixel = a * (1 - x_weight) * (1 - y_weight)+ b * x_weight * (1 - y_weight)+c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                    resized[i][j] = pixel
            resized=resized.astype(int)        
            return resized
        def bilinear_resize_1(image, height, width):      
            img_height, img_width,img_x = image.shape[:]
            resized = np.empty([height, width,img_x])
            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
            for i in range(height):
                for j in range(width):
                    x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                    x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)
                    x_weight = (x_ratio * j) - x_l
                    y_weight = (y_ratio * i) - y_l
                    a = image[y_l, x_l]
                    b = image[y_l, x_h]
                    c = image[y_h, x_l]
                    d = image[y_h, x_h]
                    pixel = a * (1 - x_weight) * (1 - y_weight)+ b * x_weight * (1 - y_weight)+c * y_weight * (1 - x_weight) + d * x_weight * y_weight
                    resized[i,j,:] = pixel
            resized=resized.astype(int)
            return resized
        p = {}
        for i in k.keys():
            if np.ndim(k[i]) == 2:
                p[i] = bilinear_resize(k[i],s*len(k[i]),s*len(k[i].T))
            else:
                p[i] = bilinear_resize_1(k[i],s*len(k[i]),s*len(k[i].T))
        return p   
        
    # Rescales the batch of input images according to a given scale . Where s is the scaling factor.
    # This function has to return a dictionary of rescaled images where the key is the name of the image and the value is corresponding rescaled output in accordance to the random indexes picked up.

    def resize(self,h,w):
        h_new=h
        w_new=w
        dic ={} 
        imgs=self._getitem_()
        for k in imgs:
            arr= imgs[k]
            height,width=arr.shape[:2]
            h_ratio=h_new/(height)
            w_ratio=w_new/(width)
            if len(arr.shape)==2:
                req = np.zeros((h_new, w_new), np.uint8)
                for i in range(h_new):
                    for j in range(w_new):
                        h_req = int(i / h_ratio)
                        w_req = int(j / w_ratio)
                        req[i, j] = arr[h_req, w_req]
            else:
                req=np.zeros((h_new,w_new,3),np.uint8)
                for i in range(h_new):
                    for j in range(w_new):
                        for t in range(3):
                            h_req = int(i / h_ratio)
                            w_req = int(j / w_ratio)
                            req[i,j,t] = arr[h_req,w_req,t]
            # Image.fromarray(req).show()
            req=req.astype(int)
            dic[k]=req
        return dic

    # Resizes the batch of input images to the given dimensions. Where h is the number of rows and w is the number of columns
    # This function has to return a dictionary of resized images where the key is the name of the image and the value is corresponding resized output in accordance to the random indexes picked up.
    # The datatype of output has to be a numpy array

    def translate(self,tx,ty):
        
        t_dic = {}
        k = self._getitem_()
        for g in k.keys(): 
            if np.ndim(k[g]) == 2:
                row,column = np.shape(k[g])
                ii = np.zeros((row + abs(ty),column + abs(tx)))
                rr = np.empty((row, column))
                if tx > 0 and ty >0:
                    ii[:row,tx:column+tx] = k[g]
                elif tx > 0 and ty < 0:
                    ii[-1*ty :row - ty ,tx:column+tx] = k[g]
                elif tx < 0 and ty >0:
                    ii[:row,:column] = k[g]    
                elif tx < 0 and ty < 0:
                    ii[-1*ty:-1*ty+row,:column] = k[g]
                else:
                    ii = k[g]
                rr = ii[:row,:column]
                rr=rr.astype(int)
                t_dic=rr
            else:
                row,column,height = np.shape(k[g])
                ii = np.zeros((row + abs(ty),column + abs(tx),height))
                rr = np.empty((row, column, height))
                a_flat=np.empty(row*column*3)
                # ii[:row,tx:column+tx,:] = k[g]
                if tx > 0 and ty >0:
                    ii[:row,tx:column+tx,:] = k[g]
                elif tx > 0 and ty < 0:
                    ii[-1*ty :row - ty ,tx:column+tx,:] = k[g]
                elif tx < 0 and ty >0:
                    ii[:row,:column,:] = k[g]    
                elif tx < 0 and ty < 0:
                    ii[-1*ty:-1*ty+row,:column,:] = k[g]
                else:
                    ii = k[g]
                rr = ii[:row,:column,:]
                z = np.max(rr)
                a_flat=rr.reshape(row*column*3)
                # print(a_flat.shape)
                for ele in a_flat:
                    if ele>255:
                        ele=ele*(255/z)
                rr=a_flat.reshape(row,column,3)
                rr=rr.astype(int)
                # plt.imshow(rr)
                # plt.show()       
                t_dic[g]=rr
                
        return t_dic                                                                                             



    # Translates the batch of input images. Where tx is the translation in x direction and ty is the translation in the y direction.
    # This function has to return a dictionary of translated images where the key is the name of the image and the value is corresponding translated output in accordance to the random indexes picked up.

    def crop(self,id1,id2,id3,id4):
        crop_dic={}
        imgdir=self._getitem_()
        for k in imgdir:
            im=imgdir[k]
        # image = plt.imread(r'C:\Users\Madke\Pictures\me.jpg')
        # im = np.asarray(image)
        if(len(im.shape)==2):
            arr = im[id1[0]:(id2[0]+1),id1[1]:(id4[1]+1)]
            crop_dic[k]=arr.astype(int)
        else:
            arr = im[id1[0]:(id2[0]+1),id1[1]:(id4[1]+1),:3]
            crop_dic[k]=arr.astype(int)
        return crop_dic    
        # plt.imshow(arr)
        # plt.show()

        


    # Crops the batch of input images to the given dimensions. Where id1,id2,id3,id4 are tuples of the indexes where cropping has to be performed
#   id1 .----------------------------------------.(x,y) = id2
    #   |                                        |
    #   |                                        |
    #   |                                        |
# id4   .----------------------------------------. id3
    # This function has to return a dictionary of cropped images where the key is the name of the image and the value is corresponding cropped output in accordance to the random indexes picked up.

    def blur(self):
        data = self._getitem_()
        key = list(data.keys())
        imgs = list(data.values())
        new_imgs = []
        for img in imgs:
            if img.ndim == 2:
                frame = np.asarray(img)
                # print(frame.shape)
                q = np.copy(frame)
                l = np.copy(frame)
                for _ in range(1):
                    l = q
                    o = np.zeros((1, len(frame.T)))
                    l = np.append(o, q, axis=0)
                    l = np.append(l, o, axis=0)
                    l = l.T
                    a = len(l.T)
                    p = np.zeros((1, a))
                    l = np.append(p, l, axis=0)
                    l = np.append(l, p, axis=0)
                    l = l.T
                    for i in range(1, len(l) - 1):
                        for j in range(1, len(l.T) - 1):
                                q[i - 1][j - 1] = (l[i][j] + l[i - 1][j] + l[i + 1][j] + l[i][j - 1] + l[i][j + 1] + l[i-1][j-1] + l[i+1][j-1] + l[i-1][j+1] + l[i+1][j+1])/9
                        # plt.imshow(q,cmap='gray')
                        # plt.show() 
                        q=q.astype(int)
                        new_imgs.append(q)
            elif img.ndim == 3:
                new_img = np.zeros((img.shape[0], img.shape[1], 3))
                for i in range(1,img.shape[0]-1):
                    for j in range(1,img.shape[1]-1):
                        #num_list = img[i-1:i+2, j-1:j+2]
                        for k in range(3):
                            median = np.median(img[i-1:i+2, j-1:j+2, k])
                            new_img[i,j,k] = median
                new_img=new_img.astype(int)
                # plt.imshow(new_img)
                # plt.show()
                new_imgs.append(new_img)
        return dict(zip(key, new_imgs))
    # Blurring the batch of input images in accordance to the filter specified.
    # This function has to return a dictionary of blurred images where the key is the name of the image and the value is corresponding blurred output in accordance to the random indexes picked up.

    def edge_detection(self):
        graydic = self.rgb2gray()
        edge_dic={}
        imgs = graydic
        for k in imgs:
            image = imgs[k]
        # image = plt.imread(r'C:\Users\Madke\Downloads\ass_4 (1)\ass_4\dc_metro.png')
            frame = np.asarray(image)
            frame = image
            q = np.copy(frame)
            l = np.copy(frame)
            r = np.copy(frame)
        #     print(frame,len(frame))
            for _ in range(1):
                l = r
                l = q
                o = np.zeros((1, len(frame.T)),dtype=int)
                l = np.append(o, q, axis=0)
                l = np.append(o, l, axis=0)
                l = np.append(l, o, axis=0)
                l = np.append(l, o, axis=0)
                l = l.T
                a = len(l.T)
                p = np.zeros((1, a),dtype=int)
                l = np.append(p, l, axis=0)
                l = np.append(p, l, axis=0)
                l = np.append(l, p, axis=0)
                l = np.append(l, p, axis=0)
                l = l.T
                for i in range(2, len(l) - 2):
                    for j in range(2, len(l.T) - 2):
                        q[i - 2][j - 2] = ((l[i][j - 1]*(-2)) + (l[i][j + 1]*(2)) + (l[i-1][j-1]*(-1)) + (l[i+1][j-1]*(-1)) + (l[i-1][j+1]*1) + (l[i+1][j+1]*1))
                        r[i - 2][j - 2] = ((l[i-1][j]*(2)) + (l[i+1][j-1]*(-1)) + (l[i-1][j-1]*(1)) + (l[i+1][j]*(-2)) + (l[i-1][j+1]*1) + (l[i+1][j+1]*(-1)))

            b = np.full([len(q),len(q.T)], 0)
        # print(b.shape)
            for i in range(len(q)):
                for j in range(len(q.T)):
                    b[i][j] = round(math.sqrt(((q[i][j])**2)+((r[i][j])**2)))
        # plt.imshow(b,cmap='gray')
        # plt.show()
            z = np.max(b)
            for i in range(b.shape[0]):
                for j in range(b.shape[1]):
                    if(b[i][j]>255):
                        b[i][j]=b[i][j]*(255/z)
            # plt.imshow(b.astype(int))
            # plt.show()            
            edge_dic[k]=b.astype(int)
        # h = list(edge_dic.values())
        # for s in h:
        #     plt.imshow(s,cmap='gray')
        #     plt.show()
        return(edge_dic)    
     
        

        # This function is to fetch items specified at the given location. Keeping in mind the batch_size, shuffle argument and the dataset location specified. It will pick up number of images = batch size and return the images.
# If shuffle = True it has to pick up random indices and if it is false it will pick up the images sequentially.
    # This function has to return a dictionary of input images where the key is the name of the image and the value is corresponding input image in accordance to the indexes picked up.
# Every other function will call the __getitem__ function to get the input images on which the operation has to be performed.
    def rgb2gray(self):
        rgb2gray_dic={}
        imgdata=self._getitem_()
        for k in imgdata:
            image=imgdata[k]
        # image = plt.imread(r'C:\Users\Madke\Pictures\Screenshots\Screenshot (143).png')
            if(len(image.shape)==3):
            # im = np.asarray(image)
            # fig, ax = plt.subplots() 
            # ax.imshow(image)
                r,g,b = image[:,:,0],image[:,:,1],image[:,:,2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                rgb2gray_dic[k]=gray.astype(int)
        #     plt.imshow(gray,cmap='gray')
        # # return gray
        #     plt.show()
            else:
                rgb2gray_dic[k]=image
                # plt.imshow(image)
                # plt.show()   
        return rgb2gray_dic          

    # Converts a batch of rgb image to a gray image.
    # This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.

    def rotate(self,theta):
        # image = np.array(plt.imread(r'C:\Users\Madke\Downloads\ir_test (1)\ir_test\me.jpg'))
        image_loc_dict = self._getitem_()
        key = list(image_loc_dict.keys())
        imgs = list(image_loc_dict.values())
        new_imgs = []
        for loc in imgs :
            image = loc
            if(len(image.shape)==2):
                theta=math.radians(theta)                               
                cosine=math.cos(theta)
                sine=math.sin(theta)
                h=image.shape[0]                                  
                w=image.shape[1] 
                original_ch = round(((image.shape[0]+1)/2)-1)   
                original_cw = round(((image.shape[1]+1)/2)-1) 
                new_h=round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
                new_w=round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1
                new_ch= round(((new_h+1)/2)-1)       
                new_cw= round(((new_w+1)/2)-1)
                output = np.zeros((new_h,new_w))
                for i in range(h):
                    for j in range(w):
                        y=image.shape[0]-1-i-original_ch                
                        x=image.shape[1]-1-j-original_cw   
                        new_y=round(-x*sine+y*cosine)
                        new_x=round(x*cosine+y*sine)
                        new_y=new_ch-new_y
                        new_x=new_cw-new_x
                        if 0 <= new_x < new_w and 0 <= new_y < new_h:
                            output[new_y,new_x]=image[i,j]                      
                # plt.imshow(output)
                # plt.show()
            if(len(image.shape)==3):
                theta=math.radians(theta)                               
                cosine=math.cos(theta)
                sine=math.sin(theta)
                height=image.shape[0]                                   
                width=image.shape[1]                                   
                new_height  = round(abs(image.shape[0]*cosine)+abs(image.shape[1]*sine))+1
                new_width  = round(abs(image.shape[1]*cosine)+abs(image.shape[0]*sine))+1
                output=np.zeros((new_height,new_width,image.shape[2]))
                original_centre_height   = round(((image.shape[0]+1)/2)-1)   
                original_centre_width    = round(((image.shape[1]+1)/2)-1) 
                new_centre_height= round(((new_height+1)/2)-1)        
                new_centre_width= round(((new_width+1)/2)-1)          
                for i in range(height):
                    for j in range(width):
                        #co-ordinates of pixel with respect to the centre of original image
                        y=image.shape[0]-1-i-original_centre_height                   
                        x=image.shape[1]-1-j-original_centre_width                      
                        #co-ordinate of pixel with respect to the rotated image
                        new_y=round(-x*sine+y*cosine)
                        new_x=round(x*cosine+y*sine)
                        new_y=new_centre_height-new_y
                        new_x=new_centre_width-new_x

                        # adding if check to prevent any errors in the processing
                        if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                            output[new_y,new_x,:]=image[i,j,:]      
            # plt.imshow(output.astype(int))
            # plt.show()
            new_imgs.append(output.astype(int))
        return dict(zip(key, new_imgs))      

    # # Rotates the batch of input images to the given dimensions. Where theta is the angle of rotation.
    # This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.

# inputarg = preprocess(r'C:\Users\Madke\Pictures\New folder',2,shuffle = False)
# print(inputarg.rescale(4))
# print(inputarg.rgb2gray())
# print(inputarg.crop((50,14),(365,14),(50,200),(365,200)))
# print(inputarg.blur())
# print(inputarg.rotate(30))
# print(inputarg.resize(119,164))
# print(inputarg.edge_detection())
# print(inputarg.translate(2,-4))
