from PIL import Image
import numpy as np
import math as math
import os
from sklearn.svm import SVC
import pandas as pd

def createArr(val, limit):
    rez = []
    for i in range(0, limit):
        rez.append(val)
    
    return rez

def genPixelAvg(filename,size,file_path):    
    img = Image.open(file_path+filename)
    if(len(size) == 2):
        img = img.resize(size)
    
    if(img.mode == 'RGB' or img.mode == 'RGBA') : 
        # pixel_values = list(img.getdata())
        avg_matrix = np.zeros(img.size) 
        for i in range(0, img.size[0]):
            for j in range(0, img.size[1]):
                sum = 0
                for k in range (0,3): 
                    sum += img.getpixel((i,j))[k]
                sum = sum/3
                avg_matrix[i][j] = sum

    return avg_matrix

def genCells(m):
    x = math.floor(len(m[0])/4)
    y = math.floor(len(m)/4)
    rez = []
    for i in range(0,y):
         rez.append([])
         for j in range(0, x):
             rez[i].append([])
             cell = []
             for k in range(0,4):
                cell.append([])
                for l in range(0,4):
                    cell[k].append(m[i*4+k][j*4+l])
             rez[i][j] = cell    
    return rez

def genHOG(file_name, file_path, matrix = []):
    if(len(matrix) == 0):
        m = genCells(genPixelAvg(file_name,(32,64), file_path+'/'))
    else:
        m = genCells(matrix)
    rez = []
    
    for i in range(0, len(m)):
        for j in range(0, len(m[0])):
            h = createArr(val=0, limit=9)
            for k in range(0, 4):
                for l in range(0,4):
                    if(k == 0): 
                        a = m[i][j][k][l]
                        b = m[i][j][k+1][l]
                    elif(k == 3):
                        a = m[i][j][k-1][l]
                        b = m[i][j][k][l]
                    else:
                        a = m[i][j][k-1][l]
                        b = m[i][j][k+1][l]

                    if(l == 0):
                        c = m[i][j][k][l]
                        d = m[i][j][k][l+1]
                    elif(l == 3):
                        c = m[i][j][k][l-1]
                        d = m[i][j][k][l]
                    else:
                        c = m[i][j][k][l-1]
                        d = m[i][j][k][l+1]

                    gx = b - a
                    gy = c - d

                    magnitude = math.sqrt(gx**2 + gy**2)    
                    
                    if(gx != 0):
                       orientation = math.atan(gy/gx)
                    else:
                        orientation = 90
                    

                    first_bin =math.floor(orientation/20)
                    second_bin = first_bin + 1

                    
                    first_bin_degree = first_bin*20
                    second_bin_degree = second_bin*20
                   
                    if(first_bin < 8):
                        h[first_bin] += math.floor((second_bin_degree-orientation)/20 * magnitude)
                        h[second_bin] += math.floor((orientation - first_bin_degree)/20 * magnitude)
                    else:
                        h[8] += magnitude

            rez.extend(h)
    
    return rez

def generateHOG(add_to_existing, signs_path_arr= [], other_path_arr = []): 
    hogs = []
    classes = []

    csv_exists = False
    for file in os.listdir():
        if(file == 'hog.csv'):
            csv_exists = True
            break

    if(csv_exists):    
        hogs = pd.read_csv('./hog.csv').values.tolist()
        classes = np.squeeze(np.asarray(pd.read_csv('./classes.csv').values)).tolist()   


    # else:
    if(not csv_exists or add_to_existing): 
        for path in signs_path_arr:
            count_files = 0
            signal_files = os.listdir(path)
            for file in signal_files:
                if(count_files >= 10):
                    break
                hogs.append(genHOG(file, path))
                classes.append(1)
                count_files += 1

        for path in other_path_arr:
            other_files = os.listdir(path)
            for file in other_files:
                hogs.append(genHOG(file, path))
                classes.append(0)    

        
        df = pd.DataFrame(hogs)
        df.to_csv('hog.csv', index=False)

        class_df = pd.DataFrame(classes)
        class_df.to_csv('classes.csv', index=False)

    
    return [hogs, classes]

def generateModel(use_existing, hogs=[], classes=[]):
    model_exists = False
    clf = object
    if(use_existing):
        for file in os.listdir():
            if(file == 'pickle'):
                model_exists = True
                break
        if(model_exists):
            clf = pd.read_pickle(filepath_or_buffer="./pickle")
    
    if(not use_existing or not model_exists):    
        clf = SVC(kernel='rbf', random_state=0)
        clf.fit(hogs,classes)
        pd.to_pickle(obj=clf,filepath_or_buffer="./pickle")

    return clf
def subMatrix(m, start, end):
    rez = []
    for i in range (start[0], end[0]):
        rez.append([])
        for j in range(start[1], end[1]):
            rez[len(rez) - 1].append(m[i][j])

    return rez
        
def findSigns(image_name, image_path, model):
    m = genPixelAvg(filename = image_name, size=[], file_path = image_path)
    col = math.floor(len(m[0]) / 32)
    row = math.floor(len(m) / 64)
    for i in range(0, row):
        for j in range(0, col):
            # print("Col: ",i*32," Row: ",j*6)
            mi = subMatrix(m,(i*64,j*32), (i*64 + 64, j*32 + 32))
            # print("Col: ",len(mi[0])," Row: ", len(mi))
            hog_to_predict = genHOG('','',matrix=mi)
            if(model.predict([hog_to_predict])):
                return [i,j]
           
   
    
    


img = Image.open('./Data/Testing/image_7.jpg')
(img.crop((0,0,64,32))).show()

# path_arr = []
# start = 100
# end = 205
# count = 0
# ls_dir = os.listdir('../Data_images/Train')
# num_ls_dir = list(map(int, ls_dir))
# num_ls_dir.sort()

# ls_dir = list(map(str, num_ls_dir))

# for path in ls_dir:
#     if(count >= end):
#         break

#     if(count >= start and count < end):
#         str = '../Data_images/Train/'+ path
#         path_arr.append(str)
    
#     count += 1

# print(path_arr, 'Folders')


# generateHOG(add_to_existing = True, signs_path_arr=path_arr, other_path_arr=['./Data/Other'])
# generateHOG(add_to_existing = True, signs_path_arr=path_arr, other_path_arr=[])


# arr = generateHOG(add_to_existing=False)

# clf = generateModel(use_existing= True ,hogs = arr[0], classes = arr[1])

clf = generateModel(use_existing=True)
findSigns('image_7.jpg','./Data/Testing/', model=clf)
# imageToPredict = [genHOG('image_10.jpg','./Data/Testing/')]
# print("Predict",clf.predict(imageToPredict)) 


