from PIL import Image
import cv2
import PyPDF2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import imutils


plt.close('all')
cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

import time
start = time.time()

# Read pdf and converting each page into images--------------------------------------------


pdfFileObj = open('R_Roche_Sample3.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj,strict=False)
numpage = pdfReader.numPages


from wand.image import Image as Img
with Img(filename='R_Roche_Sample3.pdf', resolution=300) as img:
    img.save(filename='s.jpg')

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

# Data extraction process on each page wise--------------------------------------


for ii in range(1,numpage+1):
    
    pgno = 'page'+str(ii)
    print(pgno)
    fnam = 's.jpg'
    fnam2 = fnam[0]+'-'+str(ii-1)+fnam[1:5]
    
    img = cv2.imread(fnam2)
    
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    cv2.imshow("image",img)
#    cv2.waitKey(100)


    ht,wd,dim = img.shape
    
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding the image
    (thresh, imgb) = cv2.threshold(imgray, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    
#    cv2.namedWindow('binary image',cv2.WINDOW_NORMAL)
#    cv2.imshow("binary image",imgb)
#    cv2.waitKey(100)
    
    
    # Invert the image
    imgbin = 255-imgb 
    
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('image', 700,900)
#    cv2.imshow("image", imgbin)
#    cv2.waitKey(100)
    
    
    # Defining a kernel length
    kernel_length = np.array(imgbin).shape[1]//80
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    
    
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(imgbin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
#    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
#    
#    cv2.namedWindow('verticle_lines_img',cv2.WINDOW_NORMAL)
#    cv2.imshow("verticle_lines_img",verticle_lines_img)
#    cv2.waitKey(100)
    
    
    
    strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    verticle_dil = cv2.dilate(verticle_lines_img,strl,iterations = 1)
    
#    cv2.namedWindow('verticle_dil',cv2.WINDOW_NORMAL)
#    cv2.imshow("verticle_dil",verticle_dil)
#    cv2.waitKey(100)
    
    
    kernel = np.ones((300,1), np.uint8) 
    verticle_comb = cv2.dilate(verticle_dil,kernel,iterations = 1)
    
#    cv2.namedWindow('verticle_comb',cv2.WINDOW_NORMAL)
#    cv2.imshow("verticle_comb",verticle_comb)
#    cv2.waitKey(100)
    
    
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(imgbin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
#    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
#    
#    cv2.namedWindow('horizontal_lines_img',cv2.WINDOW_NORMAL)
#    cv2.imshow("horizontal_lines_img",horizontal_lines_img)
#    cv2.waitKey(100)
    
    strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    horizontal_dil = cv2.dilate(horizontal_lines_img,strl,iterations = 1)
    
#    cv2.namedWindow('horizontal_dil',cv2.WINDOW_NORMAL)
#    cv2.imshow("horizontal_dil",horizontal_dil)
#    cv2.waitKey(100)
#    cv2.imwrite("horizontal_dil.jpg",horizontal_dil)
    
    
    
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_final_bin = 255-img_final_bin 
#    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    
    
    
    nz = cv2.countNonZero(img_final_bin);    #   finding total pixels of lines 
    
    nzver = cv2.countNonZero(verticle_lines_img)
    nzhor = cv2.countNonZero(horizontal_lines_img)
    
    ntot = ht*wd      # total pixels of image
    
    nper = nz*100/ntot
    
    if nper>1 and nzver!=0 and nzhor!=0:
        print('Table exists')
    else:
        print('No table')  
    
# If table exists then table wise extraction------------------------    
    
    if nper>1 and nzver!=0 and nzhor!=0:
    
    
        # table corners extraction----------------------
        loc1 = np.argwhere(verticle_lines_img == 255)
        
        locx = loc1[:,0]
        
        xmin = min(locx)
        xmax = max(locx)
        
        locy = loc1[:,1]
        
        ymin = min(locy)
        ymax = max(locy)
        
        # table portion-----------------------------------
        imgcrop = img[xmin:xmax,ymin:ymax]
#        cv2.namedWindow('extractedtable',cv2.WINDOW_NORMAL)
#        cv2.resizeWindow('extractedtable', 500,600)
#        cv2.imshow("extractedtable",imgcrop)
#        cv2.waitKey(100)
#        
#        cv2.imwrite("extractedtable.jpg",imgcrop)
        
        
        
        # top portion of table-------------------------------
        imgtop = img[0:xmin,:]
#        cv2.namedWindow('imgtop',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('extractedtable', 500,600)
#        cv2.imshow("imgtop",imgtop)
#        cv2.waitKey(100)
#        cv2.imwrite("imgtop.jpg",imgtop)
        
#        print('top data\n')
        txttop = pytesseract.image_to_string(imgtop)
#        print(txttop)
        
        outname = 'text-'+str(ii)+'.txt'
        file = open(outname,'w',encoding="utf-8") 
        file.write(txttop) 
        file.close() 
        
        
        # bottom portion of table----------------------------
        imgbot = img[xmax:ht,:]
#        cv2.namedWindow('imgbot',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('extractedtable', 500,600)
#        cv2.imshow("imgbot",imgbot)
#        cv2.waitKey(100)
#        cv2.imwrite("imgbot.jpg",imgbot)
        
#        print('bottom data\n')
        txtbot = pytesseract.image_to_string(imgbot)
#        print(txtbot)
        
        file = open(outname,'a',encoding="utf-8") 
        file.write('\n\n\n\n\n')
        file.write(txtbot) 
        file.close() 
            
            
        # Finding number of tables--------------------------
        
        comb_dil = verticle_dil+horizontal_dil
#        cv2.namedWindow('comb_dil',cv2.WINDOW_NORMAL)
#        cv2.imshow("comb_dil",comb_dil)
#        cv2.waitKey(100)
        
        
        cnts = cv2.findContours(comb_dil, cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        
        
        blobs = cnts[1]
        
        lsarea = []
        for j in range(0,len(blobs)):
            
            cl = blobs[j]
            area = cv2.contourArea(cl)
            lsarea.append(area)
            
        lsarea = np.float64(lsarea)
        max10 = ntot*.05
        
        
        lsind = []
        for j in range(0,len(blobs)):
            
            cl = blobs[j]
            area = cv2.contourArea(cl)
            if area < max10:
                lsind.append(j)
        lsind = tuple(lsind)
                
        blobs = [v for i, v in enumerate(blobs) if i not in lsind] 
        
        
        if len(blobs)>1:
            print('Multiple Tables')
            
# If single table, table extraction-----------------------------------------
            
        if len(blobs)<2:    
            
            
            # Removing very small vertical lines-------------------------------------
            
            
            cnts = cv2.findContours(verticle_comb, cv2.RETR_EXTERNAL,
            	cv2.CHAIN_APPROX_SIMPLE)
            
            cline = cnts[1]
            
            lsarea = []
            for j in range(0,len(cline)):
                
                cl = cline[j]
                area = cv2.contourArea(cl)
                lsarea.append(area)
                
            lsarea = np.float64(lsarea)
            maxarea = np.max(lsarea)
            max30 = maxarea*.3
            
            
            lsind = []
            for j in range(0,len(cline)):
                
                cl = cline[j]
                area = cv2.contourArea(cl)
                if area < max30:
                    lsind.append(j)
            lsind = tuple(lsind)
                    
            cline = [v for i, v in enumerate(cline) if i not in lsind]      
            
            
            clen = len(cline)
            
            df = pd.DataFrame()
            
            vimg = img.copy()
            
            # vertical lines arranged in order----------------
            
            ls = []
            
            for j in range(0,clen):
                
                cl = cline[j]
                
                # determine the most extreme points along the contour
                extLeft = tuple(cl[cl[:, :, 0].argmin()][0])
                yy = extLeft[0]
                ls.append(yy)
                
            ls = np.float64(ls)
            lsort = np.argsort(ls)
            lsort = lsort[1:]
            
            
            # vertical lines removal------------------ 
               
            loc2 = np.argwhere(verticle_dil == 255)
            for i in range(0,len(loc2)):
                
                loci = loc2[i]
                locix = loci[0]
                lociy = loci[1]
                imgb[locix,lociy] = 255
                
            #  Segmenting vertical columns------------------------------   
                
            for i in lsort:
                
                cl = cline[i]
                
                
                # determine the most extreme points along the contour
                extLeft = tuple(cl[cl[:, :, 0].argmin()][0])
                extRight = tuple(cl[cl[:, :, 0].argmax()][0])
                extTop = tuple(cl[cl[:, :, 1].argmin()][0])
                extBot = tuple(cl[cl[:, :, 1].argmax()][0])
                
                pt = (extLeft[0],extLeft[1])
                
                
                cv2.drawContours(vimg, [cl], -1, (0, 255, 255), 2)
                cv2.circle(vimg, extLeft, 24, (0, 0, 255), -1)
                cv2.circle(vimg, extRight, 24, (0, 255, 0), -1)
                cv2.circle(vimg, extTop, 24, (255, 0, 0), -1)
                cv2.circle(vimg, extBot, 24, (255, 255, 0), -1)
                 
                # show the output image
                cv2.namedWindow('Image new',cv2.WINDOW_NORMAL)
                cv2.imshow("Image new", vimg)
                cv2.waitKey(100)
            
            
                x1 = extLeft[0]
                y1 = extLeft[1]
                x2 = extBot[0]
                y2 = extBot[1]
            
            
                img1 =   imgb[xmin:y2, ymin+3:x1-3]
                img2 = 255-img1
                
#                cv2.namedWindow('Segment',cv2.WINDOW_NORMAL)
#                cv2.imshow("Segment", img1)
#                cv2.waitKey(100)
#                cv2.imwrite(iname,img1)
                
                
                # extracting horizontal lines in each column----------------------------
                
                # Morphological operation to detect horizontal lines from an image
                img_temp2 = cv2.erode(img2, hori_kernel, iterations=3)
                horizontal_column = cv2.dilate(img_temp2, hori_kernel, iterations=3)
#                cv2.imwrite("horizontal_column.jpg",horizontal_column)
                
#                cv2.namedWindow('horizontal_column',cv2.WINDOW_NORMAL)
#                cv2.imshow("horizontal_column",horizontal_column)
#                cv2.waitKey(100)
                
                strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
                horizontal_column_dil = cv2.dilate(horizontal_column,strl,iterations = 1)
                
#                cv2.namedWindow('horizontal_column_dil',cv2.WINDOW_NORMAL)
#                cv2.imshow("horizontal_column_dil",horizontal_column_dil)
#                cv2.waitKey(100)
#                cv2.imwrite("horizontal_column_dil.jpg",horizontal_column_dil)
            
                
                
                kernel = np.ones((1,500), np.uint8) 
                horizontal_comb = cv2.dilate(horizontal_column_dil,kernel,iterations = 1)
                
#                cv2.namedWindow('horizontal_comb',cv2.WINDOW_NORMAL)
#                cv2.imshow("horizontal_comb",horizontal_comb)
#                cv2.waitKey(100)
#                cv2.imwrite("horizontal_comb.jpg",horizontal_comb)
                
                
                # finding horizontal lines------------------------------------
                
                cnts1 = cv2.findContours(horizontal_comb, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
                
                cline1 = cnts1[1]
                
                # removing small horizontal lines--------------------------------------
            
                lsarea = []
                for j in range(0,len(cline1)):
                    
                    cl = cline1[j]
                    area = cv2.contourArea(cl)
                    lsarea.append(area)
                    
                lsarea = np.float64(lsarea)
                maxarea = np.max(lsarea)
                max30 = maxarea*.3
                
                
                lsind = []
                for j in range(0,len(cline1)):
                    
                    cl = cline1[j]
                    area = cv2.contourArea(cl)
                    if area < max30:
                        lsind.append(j)
                lsind = tuple(lsind)
                        
                cline1 = [v for i, v in enumerate(cline1) if i not in lsind]   
                
                clen1 = len(cline1)
                
                
                # horizontal lines removal------------------ 
               
                loc2 = np.argwhere(horizontal_comb == 255)
                for i in range(0,len(loc2)):
                    
                    loci = loc2[i]
                    locix = loci[0]
                    lociy = loci[1]
                    img1[locix,lociy] = 255
            
            
                # Cell cropping based on horizontal lines----------------
                
                ls = []
                
                for j in range(0,clen1):
                    
                    cl = cline1[j]
                    
                    # determine the most extreme points along the contour
                    extTop = tuple(cl[cl[:, :, 1].argmin()][0])
                    yy = extTop[1]
                    ls.append(yy)
                ls.reverse()
            
                
                lstxt = []
                for k in range(0,len(ls)-1):
                    
                    x11 = ls[k]
                    x12 = ls[k+1]
                
                    imgbox = img1[x11:x12,:]      # detected cell image
                    
#                    cv2.namedWindow('box',cv2.WINDOW_NORMAL)
#                    cv2.imshow("box",imgbox)
#                    cv2.waitKey(100)
#                    cv2.imwrite(iname,imgbox)
                    
                    
                    txt = pytesseract.image_to_string(imgbox)
#                    print(txt)
                
                    lstxt.append(txt)
                lstxt = pd.Series(lstxt)   
                
                
                ymin = x1
                    
                        
                df = df.append(lstxt,ignore_index=True)
                
            df = df.T    # all data combined into single dataframe
#            print(df)
            
        
            outname = 'table-'+str(ii)+'.csv'
            df.to_csv(outname)        
            

# If multiple tables , segment each table and then single table extraction same---------------------------------            
            
        else:    
    
            # Detecting each tables-------------------------------

            lenblo = len(blobs)
            
            for jj in range(0,lenblo):
            
                jj2 = lenblo-1-jj
                
            
                mask = np.zeros(comb_dil.shape, np.uint8)
            #    largest_areas = sorted(blobs, key=cv2.contourArea)
                
                cv2.drawContours(mask, [blobs[jj2]], 0, (255,255,255,255), -1)
#                cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
#                cv2.imshow("mask",mask)
#                cv2.waitKey(100)
                
                mask2 = 255-mask
#                cv2.namedWindow('mask2',cv2.WINDOW_NORMAL)
#                cv2.imshow("mask2",mask2)
#                cv2.waitKey(100)
                
                
                imgb2 = cv2.multiply(imgb, mask)
        #        cv2.namedWindow('binary image',cv2.WINDOW_NORMAL)
        #        cv2.imshow("binary image",imgb)
        #        cv2.waitKey(100)
                
                imgbin = 255-imgb2 
        #        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        #        cv2.resizeWindow('image', 700,900)
        #        cv2.imshow("image", imgbin)
        #        cv2.waitKey(100)
                
                
                imgbin = cv2.multiply(imgbin, mask)
#                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#                cv2.imshow("image",imgbin)
#                cv2.waitKey(100)
                
                imgb2 = 255-imgbin 
#                cv2.namedWindow('binary image',cv2.WINDOW_NORMAL)
#                cv2.resizeWindow('binary image', 700,900)
#                cv2.imshow("binary image", imgb2)
#                cv2.waitKey(100)
                
                
                
                # Defining a kernel length
                kernel_length = np.array(imgbin).shape[1]//80
                 
                # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
                verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
                hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                # A kernel of (3 X 3) ones.
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                
                
                
                # Morphological operation to detect vertical lines from an image
                img_temp1 = cv2.erode(imgbin, verticle_kernel, iterations=3)
                verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
#                cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
                
#                cv2.namedWindow('verticle_lines_img',cv2.WINDOW_NORMAL)
#                cv2.imshow("verticle_lines_img",verticle_lines_img)
#                cv2.waitKey(100)
                
                
                
                strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
                verticle_dil = cv2.dilate(verticle_lines_img,strl,iterations = 1)
                
#                cv2.namedWindow('verticle_dil',cv2.WINDOW_NORMAL)
#                cv2.imshow("verticle_dil",verticle_dil)
#                cv2.waitKey(100)
                
                kernel = np.ones((500,1), np.uint8) 
                verticle_comb = cv2.dilate(verticle_dil,kernel,iterations = 1)
                
#                cv2.namedWindow('verticle_comb',cv2.WINDOW_NORMAL)
#                cv2.imshow("verticle_comb",verticle_comb)
#                cv2.waitKey(100)
                
                
                # Morphological operation to detect horizontal lines from an image
                img_temp2 = cv2.erode(imgbin, hori_kernel, iterations=3)
                horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
#                cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
                
#                cv2.namedWindow('horizontal_lines_img',cv2.WINDOW_NORMAL)
#                cv2.imshow("horizontal_lines_img",horizontal_lines_img)
#                cv2.waitKey(100)
                
                strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
                horizontal_dil = cv2.dilate(horizontal_lines_img,strl,iterations = 1)
                
#                cv2.namedWindow('horizontal_dil',cv2.WINDOW_NORMAL)
#                cv2.imshow("horizontal_dil",horizontal_dil)
#                cv2.waitKey(100)
#                cv2.imwrite("horizontal_dil.jpg",horizontal_dil)
                
                
                
                # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
                alpha = 0.5
                beta = 1.0 - alpha
                # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
                img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
                img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
                (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img_final_bin = 255-img_final_bin 
#                cv2.imwrite("img_final_bin.jpg",img_final_bin)
                
                
                
                # table corners extraction----------------------
                loc1 = np.argwhere(verticle_lines_img == 255)
                
                locx = loc1[:,0]
                
                xmin = min(locx)
                xmax = max(locx)
                
                locy = loc1[:,1]
                
                ymin = min(locy)
                ymax = max(locy)
                
                # table portion-----------------------------------
                imgcrop = img[xmin:xmax,ymin:ymax]
#                cv2.namedWindow('extractedtable',cv2.WINDOW_NORMAL)
#                cv2.resizeWindow('extractedtable', 500,600)
#                cv2.imshow("extractedtable",imgcrop)
#                cv2.waitKey(100)
#                
#                cv2.imwrite("extractedtable.jpg",imgcrop)
                
                
                
                
                # Removing very small  vertical lines-------------------------------------
                
                
                cnts = cv2.findContours(verticle_comb, cv2.RETR_EXTERNAL,
                	cv2.CHAIN_APPROX_SIMPLE)
                
                cline = cnts[1]
                
                lsarea = []
                for j in range(0,len(cline)):
                    
                    cl = cline[j]
                    area = cv2.contourArea(cl)
                    lsarea.append(area)
                    
                lsarea = np.float64(lsarea)
                maxarea = np.max(lsarea)
                max30 = maxarea*.3
                
                
                lsind = []
                for j in range(0,len(cline)):
                    
                    cl = cline[j]
                    area = cv2.contourArea(cl)
                    if area < max30:
                        lsind.append(j)
                lsind = tuple(lsind)
                        
                cline = [v for i, v in enumerate(cline) if i not in lsind]      
                
                
                clen = len(cline)
                
                df = pd.DataFrame()
                
                vimg = img.copy()
                
                # vertical lines arranged in order----------------
                
                ls = []
                
                for j in range(0,clen):
                    
                    cl = cline[j]
                    
                    # determine the most extreme points along the contour
                    extLeft = tuple(cl[cl[:, :, 0].argmin()][0])
                    yy = extLeft[0]
                    ls.append(yy)
                    
                ls = np.float64(ls)
                lsort = np.argsort(ls)
                lsort = lsort[1:]
                
                
                # vertical lines removal------------------ 
                   
                loc2 = np.argwhere(verticle_dil == 255)
                for i in range(0,len(loc2)):
                    
                    loci = loc2[i]
                    locix = loci[0]
                    lociy = loci[1]
                    imgb2[locix,lociy] = 255
                    
                    
                    
                c=0
                for i in lsort:
                    
                    cl = cline[i]
                    
                    
                    # determine the most extreme points along the contour
                    extLeft = tuple(cl[cl[:, :, 0].argmin()][0])
                    extRight = tuple(cl[cl[:, :, 0].argmax()][0])
                    extTop = tuple(cl[cl[:, :, 1].argmin()][0])
                    extBot = tuple(cl[cl[:, :, 1].argmax()][0])
                    
                    pt = (extLeft[0],extLeft[1])
                    
                    
                    cv2.drawContours(vimg, [cl], -1, (0, 255, 255), 2)
                    cv2.circle(vimg, extLeft, 24, (0, 0, 255), -1)
                    cv2.circle(vimg, extRight, 24, (0, 255, 0), -1)
                    cv2.circle(vimg, extTop, 24, (255, 0, 0), -1)
                    cv2.circle(vimg, extBot, 24, (255, 255, 0), -1)
                     
                    # show the output image
#                    cv2.namedWindow('Image new',cv2.WINDOW_NORMAL)
#                    cv2.imshow("Image new", vimg)
#                    cv2.waitKey(100)
                
                
                    x1 = extLeft[0]
                    y1 = extLeft[1]
                    x2 = extBot[0]
                    y2 = extBot[1]
                
                
                    img1 =   imgb2[xmin:y2, ymin+3:x1-3]
                    img2 = 255-img1
                    
#                    cv2.namedWindow('Segment',cv2.WINDOW_NORMAL)
#                    cv2.imshow("Segment", img1)
#                    cv2.waitKey(100)
                    
                    c = c+1
                    iname = str(c)+'.jpg'
#                    cv2.imwrite(iname,img1)
                    
                    
                    # extracting horizontal lines in each column----------------------------
                    
                    # Morphological operation to detect horizontal lines from an image
                    img_temp2 = cv2.erode(img2, hori_kernel, iterations=3)
                    horizontal_column = cv2.dilate(img_temp2, hori_kernel, iterations=3)
#                    cv2.imwrite("horizontal_column.jpg",horizontal_column)
                    
#                    cv2.namedWindow('horizontal_column',cv2.WINDOW_NORMAL)
#                    cv2.imshow("horizontal_column",horizontal_column)
#                    cv2.waitKey(100)
                    
                    strl = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
                    horizontal_column_dil = cv2.dilate(horizontal_column,strl,iterations = 1)
                    
#                    cv2.namedWindow('horizontal_column_dil',cv2.WINDOW_NORMAL)
#                    cv2.imshow("horizontal_column_dil",horizontal_column_dil)
#                    cv2.waitKey(100)
#                    cv2.imwrite("horizontal_column_dil.jpg",horizontal_column_dil)
#                
                    
                    
                    kernel = np.ones((1,500), np.uint8) 
                    horizontal_comb = cv2.dilate(horizontal_column_dil,kernel,iterations = 1)
                    
#                    cv2.namedWindow('horizontal_comb',cv2.WINDOW_NORMAL)
#                    cv2.imshow("horizontal_comb",horizontal_comb)
#                    cv2.waitKey(100)
#                    cv2.imwrite("horizontal_comb.jpg",horizontal_comb)
                    
                    
                    # finding horizontal lines------------------------------------
                    
                    cnts1 = cv2.findContours(horizontal_comb, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                    
                    cline1 = cnts1[1]
                    
                    
                    
                      # removing small horizontal lines--------------------------------------
            
                    lsarea = []
                    for j in range(0,len(cline1)):
                        
                        cl = cline1[j]
                        area = cv2.contourArea(cl)
                        lsarea.append(area)
                        
                    lsarea = np.float64(lsarea)
                    maxarea = np.max(lsarea)
                    max30 = maxarea*.3
                    
                    
                    lsind = []
                    for j in range(0,len(cline1)):
                        
                        cl = cline1[j]
                        area = cv2.contourArea(cl)
                        if area < max30:
                            lsind.append(j)
                    lsind = tuple(lsind)
                            
                    cline1 = [v for i, v in enumerate(cline1) if i not in lsind]   
            
                    clen1 = len(cline1)
                    
                    
                    # horizontal lines removal------------------ 
                   
                    loc2 = np.argwhere(horizontal_comb == 255)
                    for i in range(0,len(loc2)):
                        
                        loci = loc2[i]
                        locix = loci[0]
                        lociy = loci[1]
                        img1[locix,lociy] = 255
                
                
                    # Cell cropping based on horizontal lines----------------
                    
                    ls = []
                    
                    for j in range(0,clen1):
                        
                        cl = cline1[j]
                        
                        # determine the most extreme points along the contour
                        extTop = tuple(cl[cl[:, :, 1].argmin()][0])
                        yy = extTop[1]
                        ls.append(yy)
                    ls.reverse()
                
                    
                    lstxt = []
                    d=0
                    for k in range(0,len(ls)-1):
                        
                        x11 = ls[k]
                        x12 = ls[k+1]
                    
                        imgbox = img1[x11:x12,:]
                        d=d+1
                        
#                        cv2.namedWindow('box',cv2.WINDOW_NORMAL)
#                        cv2.imshow("box",imgbox)
#                        cv2.waitKey(100)
                        iname = 'box'+str(d)+'.jpg'
#                        cv2.imwrite(iname,imgbox)
                        
                        
                        txt = pytesseract.image_to_string(imgbox)
#                        print(txt)
                    
                        lstxt.append(txt)
                    lstxt = pd.Series(lstxt)   
                    
                    
                    ymin = x1
                        
                            
                    df = df.append(lstxt,ignore_index=True)
                    
                df = df.T 
#                print(df)
                
                
                outname = 'table-'+str(ii)+'.'+str(jj)+'.csv'
                df.to_csv(outname)   
            
# If no table in page, then direct ocr application----------------------        
    else:
        
        
        txtext = pytesseract.image_to_string(imgbin)
#        print(txtext)
        
        outname = 'text-'+str(ii)+'.txt'
        file = open(outname,'w',encoding="utf-8") 
        file.write(txtext) 
        
        
end = time.time()  
tottime =  (end-start)/60     
        
# finding time for entire extraction------------

    
#print('\ntotal time: \n',tottime)


