from PIL import Image
import numpy as np
import math as math


def genPixelAvg(filename,size):    
    img = Image.open('./Assets/'+filename)
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

def genHOG(m):
    rez = []
    for i in range(0, len(m)):
        for j in range(0, len(m[0])):
            h = np.zeros(9)
            for k in range(0, 4):
                for l in range(0,4):
                    # print("i: ",i,", j: ",j,", k: ",k,", l: ",l)
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
                        h[first_bin] += (second_bin_degree-orientation)/20 * magnitude
                        h[second_bin] += (orientation - first_bin_degree)/20 * magnitude
                    else:
                        h[8] += magnitude

            rez.append(h)
    
    return rez

m = genCells(genPixelAvg('img3.jpg',(32,64)))
genHOG(m)