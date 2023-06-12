from PIL import Image
import numpy as np
import math as math
import os
from sklearn.svm import SVC
import pandas as pd

cell_size = 8

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

def genPixelAvgFromImage(img):    
    # img = Image.open(file_path+filename)
    # if(len(size) == 2):
    #     img = img.resize(size)
    
    if(img.mode == 'RGB' or img.mode == 'RGBA') : 
        # pixel_values = list(img.getdata())
        avg_matrix = np.zeros((img.size[1], img.size[0])) 
        for i in range(0, img.size[1]):
            for j in range(0, img.size[0]):
                sum = 0
                for k in range (0,3): 
                    sum += img.getpixel((j, i))[k]
                sum = sum/3
                avg_matrix[i][j] = sum

    return avg_matrix

def genCells(m):
    x = math.floor(len(m[0])/cell_size)
    y = math.floor(len(m)/cell_size)
    rez = []
    for i in range(0,y):
         rez.append([])
         for j in range(0, x):
             rez[i].append([])
             cell = []
             for k in range(0,cell_size):
                cell.append([])
                for l in range(0,cell_size):
                    cell[k].append(m[i*cell_size+k][j*cell_size+l])
             rez[i][j] = cell    
    return rez

def genHOG(file_name, file_path, matrix = []):
    if(len(matrix) == 0):
        m = genCells(genPixelAvg(file_name,(64,128), file_path+'/'))
    else:
        m = genCells(matrix)
    rez = []

    # m = [[[[121, 10], [48, 152]]]]

    for i in range(0, len(m)):
        for j in range(0, len(m[0])):
            h = createArr(val=0, limit=9)
            for k in range(0, cell_size):
                for l in range(0,cell_size):
                    if(k == 0): 
                        c = m[i][j][k][l]
                        d = m[i][j][k+1][l]
                    elif(k == cell_size - 1):
                        c = m[i][j][k-1][l]
                        d = m[i][j][k][l]
                    else:
                        c = m[i][j][k-1][l]
                        d = m[i][j][k+1][l]

                    if(l == 0):
                        a = m[i][j][k][l]
                        b = m[i][j][k][l+1]
                    elif(l ==  cell_size - 1):
                        a = m[i][j][k][l-1]
                        b = m[i][j][k][l]
                    else:
                        a = m[i][j][k][l-1]
                        b = m[i][j][k][l+1]

                    gx = b - a
                    gy = c - d

                    magnitude = math.sqrt(gx**2 + gy**2)    
                    
                    if(gx != 0):
                       orientation = 90 + math.degrees(math.atan(gy/gx))
                       
                    else:
                        orientation = 90
                    
                    # if(i == 0 and j ==0):
                    #     print("b: ",b)
                    #     print("a: ",a)
                    #     print("c: ",c)
                    #     print("d: ",d)
                    #     print("Gx: ",gx)
                    #     print("Gy: ",gy)
                    #     print("Magnitude: ", magnitude)
                    #     print("orientation: ", orientation)
                    #     return

                    first_bin =math.floor(orientation/20)
                    second_bin = first_bin + 1

                    
                    first_bin_degree = first_bin*20
                    second_bin_degree = second_bin*20
                   
                    if(first_bin < 8):
                        h[first_bin] += math.floor((second_bin_degree-orientation)/20 * magnitude)
                        h[second_bin] += math.floor((orientation - first_bin_degree)/20 * magnitude)
                    else:
                        h[8] += math.floor(magnitude)

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
    # print("Start: ",start, " End: ",end)
    rez = []
    for i in range (start[0], end[0]):
        rez.append([])
        for j in range(start[1], end[1]):
            rez[len(rez) - 1].append(m[i][j])

    return rez
        
def findSigns(image_name, image_path, model):
    main_img = Image.open(image_path+image_name)
    size = main_img.size
    step = 10
    col = math.floor(size[0] / step)
    row = math.floor(size[1] / step)
    results = []
    m = genPixelAvgFromImage(main_img)
    print("Row: ",len(m), "Col: ",len(m[0]))
    for i in range(0, row - 14):
        for j in range(0, col - 7):
            
            mi = subMatrix(m,(i * step, j * step), ( i * step + 128, j * step + 64))
            hog_to_predict = genHOG('','',matrix=mi)
           
            if(model.predict([hog_to_predict])):
                # main_img.crop((j * 10, i * 10, j * 10 + 64, i * 10 + 128)).show()
                # return [i * 10,j * 10] 
                results.append((i * step, j * step))    
                # print("Match i:",i * 10," j: ",j * 10) 

    return results        
        
    
def accuracy( signs_path_arr, other_path_arr, model):
    total_instances = 0
    count_accurate = 0
    for path in signs_path_arr:
        count_files = 0
        signal_files = os.listdir(path)

        for file in signal_files:
            
            hog = [genHOG(file, path)]
            if(model.predict(hog)[0] == 0):
                count_accurate += 1

            if(count_files > 500):
                break
            
            total_instances += 1
            count_files += 1

    for path in other_path_arr:
        count_files = 0
        other_files = os.listdir(path)

        for file in other_files:
            hog = [genHOG(file, path)]
            if(model.predict(hog)[0] == 1):
                count_accurate += 1

            if(count_files > 390):
                break

            count_files += 1
            total_instances += 1

    rez = "Accurate: ",count_accurate,". All: ",total_instances
    return rez

# img = Image.open('./Data/Testing/image_7.jpg')
# (img.crop((0,0,64,32))).show()

path_arr = []
start = 0
end = 205
count = 0
ls_dir = os.listdir('../Data_images/Train')
num_ls_dir = list(map(int, ls_dir))
num_ls_dir.sort()

ls_dir = list(map(str, num_ls_dir))

for path in ls_dir:
    if(count >= end):
        break

    if(count >= start and count < end):
        str = '../Data_images/Train/'+ path
        path_arr.append(str)
    
    count += 1


# print(path_arr, 'Folders')



# generateHOG(add_to_existing = False, signs_path_arr=path_arr, other_path_arr=['../Others_images','../Others_images_cropped'])
# generateHOG(add_to_existing = True,signs_path_arr=[], other_path_arr=['../Others_images_cropped'])


# arr = generateHOG(add_to_existing=False)
# clf = generateModel(use_existing= True ,hogs = arr[0], classes = arr[1])
# print(accuracy(signs_path_arr = [],other_path_arr=['./Images'], model= clf))
# print(accuracy(signs_path_arr = ['../Data_images/Test'],other_path_arr=[], model= clf))

clf = generateModel(use_existing=True)
print("Results: ",findSigns('image_7.jpg','./Testing/', model=clf))





# for image in os.listdir('../Others_images'):
#     img = Image.open('../Others_images/'+image)
#     img = img.crop((math.floor(img.size[0]/2) - 30, math.floor(img.size[1]/2) - 30, img.size[0] - 30, img.size[1] - 30))
#     img.save('../Others_images_cropped/'+image)

# img = Image.open('../Others_images/'+'photo-1541958409-7618fd1ad26e.jpg')
# print("??",(math.floor(img.size[0]/2) - 30, math.floor(img.size[1]/2) - 30, img.size[0] - 30, img.size[1] - 30))
# img = img.crop((math.floor(img.size[0]/2) - 30, math.floor(img.size[1]/2) - 30, img.size[0] - 30, img.size[1] - 30))
# # img.show()
# img.save('../crop_image.jpg')


###useless code maybe
    # col = math.floor(len(m[0]) / 64)
    # row = math.floor(len(m) / 128)
    # for i in range(0, row):
    #     for j in range(0, col):
            
    #         # mi = subMatrix(m,(j*64, i*128), ( i*128 + 128, j*64 + 64))
    #         mi = subMatrix(m,(i*128, j*64), ( i*128 + 128, j*64 + 64))
    #         print("X: ", len(mi[0]), " Y: ", len(mi))
    #         hog_to_predict = genHOG('','',matrix=mi)
    #         if(model.predict([hog_to_predict])):
    #             return [i,j]      