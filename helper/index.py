import numpy as np
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import generic_filter as gf
import scipy.stats.mstats as ms
from scipy.stats import skew
import scipy.ndimage.morphology as morph
from scipy import ndimage
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import random
import math
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.filters import gabor
from skimage.util import random_noise
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, precision_score
from numba import jit
from numba import prange


"""
GENERAL FUNCTIONS
"""

def calculateAndPrintResults(prediction, evaluation):
    print("points examined:")
    print(len(prediction.reshape(-1)))
    matrix = confusion_matrix(evaluation.reshape(-1), prediction.reshape(-1))
    print("Confusion matrix:")
    print(matrix)   
    print("accuracy rate:")
    print(accuracy_score(evaluation.reshape(-1), prediction.reshape(-1)))
    print("true positive rate:") 
    print(recall_score(evaluation.reshape(-1), prediction.reshape(-1)))
    print("precision rate:") 
    print(round(precision_score(evaluation.reshape(-1), prediction.reshape(-1)), 4))
    processed_kappa = cohen_kappa_score(evaluation.reshape(-1), prediction.reshape(-1))
    print("Kappa rating: ", round(processed_kappa, 4))

def extract_numpy_files_in_folder(path, skip=[]):
    """
    Returns all the .npy files in a given directory.
    Skips subdirectorys
    """
    root, _, files = next(os.walk(path))
    holder = []
    for file in files:
        if file[-3:] != ".npy":
            continue
        elif file in skip:
            continue
        holder.append(os.path.join(root,file))
    return holder

def generate_mask(small_radius, big_radius):
    """
    Generate a mask in a circular radius around a point.
    Possibly to create ring masks with this
    """
    height, width = big_radius*2,big_radius*2
    Y, X = np.ogrid[:height+1, :width+1]
    distance_from_center = np.sqrt((X- big_radius)**2 + (Y-big_radius)**2)
    mask = (small_radius <= distance_from_center) & (distance_from_center <= big_radius)
    return mask

def circel_mask_generator(radius):
    """
    Wrapper around generate_mask for usage when only a circular radius
    is required.
    """
    return generate_mask(1,radius)

def create_filter_with_mask(postfix, arr_with_filenames, function, mask):
    """
    Create a filter over an array of filenames.npy files,
    Existing files with correct naming schemas will NOT be updated if existing.
    _raw files will be skipped.
    Returns a iterator that can be used to show/save filterd array. A name is also yielded.
    """
    for filename in arr_with_filenames:
        if filename[-4:] != "_raw":
            continue
        elif os.path.isfile(f"./{filename[:-4]}_{postfix}.npy"):
            continue
        arr = np.load(f"{filename}.npy")
        holder = gf(arr, function, footprint=mask)
        yield (f"{filename[:-4]}_{postfix}", holder)

def merge_numpy_zones_files(list_of_files):
    """
    Takes a list of paths to files and load them into a panda DataFrame.
    if the name contains the word ditches the name is replaced by 'labels'
    Only one zone should be contained inside the list.
    """
    holder = {}
    for file in list_of_files:
        if "ditches" in file or "Ditches" in file:
            holder["labels"] = np.load(file).reshape(-1)
        else:
            holder [file.split("/")[-1][:-4]] = np.load(file).reshape(-1)
    return pd.DataFrame(data=holder)

def create_conic_mask(radius, direction):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    if direction == 0: #topright
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 1: #righttop
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 2: #rightbottom
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 3: #bottomright
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 4: #bottomleft
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x < 0)
    elif direction == 5: #leftbottom
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 6: #lefttop
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 7: #topleft
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x < 0)
    kernel[mask] = 1
    return kernel

def create_circular_mask(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    mask[radius][radius] = 0
    kernel[mask] = 1
    return kernel


"""------------------------------------------------------------------------------------------------------------------------------"""







"""
FEATURES/FILTERS
"""

"""
DITCHES
"""

def createBalancedMask(ditchArr, height, width):
    newArr = ditchArr.copy()
    print("in function")
    for i in range(0, len(ditchArr), height):
        for j in range(0, len(ditchArr[i]), width):
            zoneContainsDitches = None
            if (random.random() * 100 > 92.5):
                zoneContainsDitches = True
            for k in range(height):
                for l in range(width):
                    if ditchArr[i+k][j+l] == 1:
                        zoneContainsDitches = True
                    if zoneContainsDitches == True:
                        for m in range(height):
                            for n in range(width):
                                newArr[i+m][j+n] = 1
                    if zoneContainsDitches == True:
                        break
                if zoneContainsDitches == True:
                    break
            if zoneContainsDitches == None:
                for m in range(height):
                    for n in range(width):
                        newArr[i+m][j+n] = 0
    return newArr
                    


"""------------------------------------------------------------------------------------------------------------------------------"""

"""
SKYVIEWFACTOR
"""

def conicMean(arr, maskRadius, threshold):
    #Standard values: maskRadius = 5, threshold = 0.975
    masks = []
    for i in range(0, 8):
        masks.append(create_conic_mask(maskRadius, i))
    print(len(masks))
    newArr = arr.copy()
    amountOfThresholds = 0
    for i in range(len(arr)):
        print(i)
        for j in range(len(arr[i])):            
            values = meanFromMasks(arr, (i, j), masks)      
            dir1 = 2
            dir2 = 2
            dir3 = 2
            dir4 = 2     
            if values[0] < threshold and values[4] < threshold:
                dir1 = values[0] if values[0] < values[4] else values[4]
            if values[1] < threshold and values[5] < threshold:
                dir2 = values[1] if values[0] < values[5] else values[4]
            if values[2] < threshold and values[6] < threshold:
                dir3 = values[2] if values[0] < values[6] else values[4]
            if values[3] < threshold and values[7] < threshold:
                dir4 = values[3] if values[0] < values[7] else values[4]       
            dir5 = dir1 if dir1 < dir2 else dir2
            dir6 = dir3 if dir3 < dir4 else dir4    
            lowest = dir5 if dir5 < dir6 else dir6
            if lowest < threshold:
                amountOfThresholds += 1
                newArr[i][j] = 0.95 * lowest if lowest * 0.95 < arr[i][j] else arr[i][j]
    print(amountOfThresholds)        
    return newArr

def meanFromMasks(arr, index, masks):
    row = index[0]
    col = index[1]
    halfMask = len(masks[0]) // 2    
    arrLenRow = len(arr)
    arrLenCol = len(arr[row])
    values = np.zeros(8)
    elementAmounts = np.zeros(8)
    for i in range(-halfMask , halfMask):
        for j in range(-halfMask , halfMask):
            if arrLenCol > col + j + 1 and col + j + 1 >= 0 and arrLenRow > row + i + 1 and row + i + 1 >= 0:
                if masks[0][i + halfMask][j + halfMask] == 1:
                    values[0] += arr[row + i][col + j]
                    elementAmounts[0] += 1
                elif masks[1][i + halfMask][j + halfMask] == 1:
                    values[1] += arr[row + i][col + j]
                    elementAmounts[1] += 1
                elif masks[2][i + halfMask][j + halfMask] == 1:
                    values[2] += arr[row + i][col + j]
                    elementAmounts[2] += 1
                elif masks[3][i + halfMask][j + halfMask] == 1:
                    values[3] += arr[row + i][col + j]
                    elementAmounts[3] += 1
                elif masks[4][i + halfMask][j + halfMask] == 1:
                    values[4] += arr[row + i][col + j]
                    elementAmounts[4] += 1
                elif masks[5][i + halfMask][j + halfMask] == 1:
                    values[5] += arr[row + i][col + j]
                    elementAmounts[5] += 1
                elif masks[6][i + halfMask][j + halfMask] == 1:
                    values[6] += arr[row + i][col + j]
                    elementAmounts[6] += 1
                elif masks[7][i + halfMask][j + halfMask] == 1:
                    values[7] += arr[row + i][col + j]
                    elementAmounts[7] += 1
    
    for i in range(len(values)):
        values[i] = values[i] / elementAmounts[i] if elementAmounts[i] != 0 else 0.99
    return values

def skyViewNonDitchAmplification(arr):
    arr = gf(arr, np.nanmedian, footprint=create_circular_mask(25))
    newArr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.92:
                newArr[i][j] = 46
            elif arr[i][j] < 0.93:
                newArr[i][j] = 37
            elif arr[i][j] < 0.94:
                newArr[i][j] = 29
            elif arr[i][j] < 0.95:
                newArr[i][j] = 22
            elif arr[i][j] < 0.96:
                newArr[i][j] = 16
            elif arr[i][j] < 0.97:
                newArr[i][j] = 11
            elif arr[i][j] < 0.98:
                newArr[i][j] = 7
            elif arr[i][j] < 0.985:
                newArr[i][j] = 4
            elif arr[i][j] < 0.99:
                newArr[i][j] = 2
            else:
                newArr[i][j] = 1
    return gf(newArr, np.nanmean, footprint=create_circular_mask(10))

@jit
def skyViewGabor(skyViewArr):
    gabors = []
    for i in np.arange(0.03, 0.08, 0.01):
        print(i)
        for j in np.arange(0, 3, 0.52):
            gabors.append(gabor(skyViewArr, theta=j, frequency=i)[0])
    merged = skyViewArr.copy()
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            merged[i][j] = 0
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            for k in range(len(gabors)):
                merged[i][j] += gabors[k][i][j]
    return merged


"""------------------------------------------------------------------------------------------------------------------------------"""


"""
IMPOUNDMENT
"""

def impoundmentAmplification(arr):
    newArr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 0:
                newArr[i][j] = 0
            elif arr[i][j] < 1000000000:
                newArr[i][j] = 5
            elif arr[i][j] < 1010000000:
                newArr[i][j] = 50
            elif arr[i][j] < 1020000000:
                newArr[i][j] = 100
            elif arr[i][j] < 1030000000:
                newArr[i][j] = 1000
            elif arr[i][j] < 1040000000:
                newArr[i][j] = 10000
            elif arr[i][j] < 1050000000:
                newArr[i][j] = 100000
            else:
                newArr[i][j] = 1000000

    mask = create_circular_mask(10)
    return gf(gf(gf(newArr, np.nanmean, footprint=mask), np.nanmean, footprint=mask), np.nanmedian, footprint=mask)

def streamAmplification(arr):
    streamAmp = arr.copy()
    for i in range(len(streamAmp)):
        for j in range(len(streamAmp[i])):
            if streamAmp[i][j] < 14:
                streamAmp[i][j] = 0
    morphed = morph.grey_dilation(streamAmp, structure = create_circular_mask(25))
    smoothedOut = gf(morphed, np.nanmean, footprint= create_circular_mask(10))
    return smoothedOut


"""------------------------------------------------------------------------------------------------------------------------------"""


"""
HPMF
"""

def hpmfFilter(arr):
    binary = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 1000000 and arr[i][j] > -1000000:
                binary[i][j] = 1000000000
            else:
                binary[i][j] = 0
    mean = gf(gf(gf(gf(binary, np.amax, footprint=create_circular_mask(1)), np.amax, footprint=create_circular_mask(1)), np.median, footprint=create_circular_mask(2)), np.nanmean, footprint=create_circular_mask(5))
    reclassify = mean.copy()
    for i in range(len(mean)):
        for j in range(len(mean[i])):
            if mean[i][j] < 1:
                reclassify[i][j] = 0
            elif mean[i][j] < 30000000:
                reclassify[i][j] = 1
            elif mean[i][j] < 70000000:
                reclassify[i][j] = 2
            elif mean[i][j] < 100000000:
                reclassify[i][j] = 50
            elif mean[i][j] < 200000000:
                reclassify[i][j] = 75
            elif mean[i][j] < 500000000:
                reclassify[i][j] = 100
            elif mean[i][j] < 800000000:
                reclassify[i][j] = 300
            elif mean[i][j] < 1000000000:
                reclassify[i][j] = 600
            else:
                reclassify[i][j] = 1000
    return gf(reclassify, np.nanmean, footprint=create_circular_mask(7))


"""------------------------------------------------------------------------------------------------------------------------------"""



"""
SLOPE
"""

def slopeNonDitchAmplification(arr):
    newArr = arr.copy()
    arr = gf(arr, np.nanmedian, footprint=create_circular_mask(35))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 8:
                newArr[i][j] = 0
            elif arr[i][j] < 9:
                newArr[i][j] = 20
            elif arr[i][j] < 10:
                newArr[i][j] = 25
            elif arr[i][j] < 11:
                newArr[i][j] = 30
            elif arr[i][j] < 13:
                newArr[i][j] = 34
            elif arr[i][j] < 15:
                newArr[i][j] = 38
            elif arr[i][j] < 17:
                newArr[i][j] = 42
            elif arr[i][j] < 19:
                newArr[i][j] = 46
            elif arr[i][j] < 21:
                newArr[i][j] = 50
            else:
                newArr[i][j] = 55
    return gf(newArr, np.nanmean, footprint=create_circular_mask(15))

"""------------------------------------------------------------------------------------------------------------------------------"""


"""DEM"""



"""------------------------------------------------------------------------------------------------------------------------------"""
"""------------------------------------------------------------------------------------------------------------------------------"""







"""
POST-PROCESSING
"""

def find_max_distance(A):
    """
    Returns the maximum distance from  2x points
    each point is represented by a x,y cord.
    """
    #assert(A.shape[1] == 2)
    return nanmax(squareform(pdist(A)))

def get_max_distance_from_list(clusters):
    """
    Wrapper around find_max_distance
    to handle a list of 'clusters'
    """
    return list(map(find_max_distance, clusters))

def rasterToZones(arr, zoneSize, threshold):
    newArr = arr.copy()
    print(len(arr))
    print(len(arr[0]))
    for i in range(0, len(arr), zoneSize):
        for j in range(0, len(arr[i]), zoneSize):
            numberOfClassified = 0
            if i < len(arr) - zoneSize and j < len(arr[i]) - zoneSize:
                for k in range(zoneSize):
                    for l in range(zoneSize):
                        if arr[i + k][j + l] == 1:
                            numberOfClassified += 1
                if numberOfClassified > (zoneSize**2)/threshold:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            newArr[i + k][j + l] = 1
                else:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            newArr[i + k][j + l] = 0
    return newArr

def probaToZones(arr, zoneSize, threshold):
    newArr = np.empty([len(arr), len(arr[0])])
    print(newArr.shape)
    for i in range(0, len(arr), zoneSize):
        for j in range(0, len(arr[i]), zoneSize):
            totalProba = 0
            if i < len(arr) - zoneSize and j < len(arr[i]) - zoneSize:
                for k in range(zoneSize):
                    for l in range(zoneSize):
                        totalProba += arr[i+k][j+l]
                if totalProba / zoneSize**2 > threshold:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            newArr[i + k][j + l] = 1
                else:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            newArr[i + k][j + l] = 0
    return newArr

def customRemoveNoise(arr, radius, threshold, selfThreshold):
    newArr = arr.copy()
    print("creating maxArr")
    maxArr = gf(arr, np.nanmax, footprint=create_circular_mask(radius))
    print("maxArr created")
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if maxArr[i][j] < threshold and arr[i][j] < selfThreshold:
                newArr[i][j] *= 0.25
    return newArr

def probaNoiseReduction(arr):
    deNoise15 = denoise_bilateral(arr, sigma_spatial=15, multichannel=False)
    deNoiseStepTwo = customRemoveNoise(deNoise15, 10, 0.9, 0.5)
    return deNoiseStepTwo


def probaPostProcess(arr, zoneSize, probaThreshold):
    deNoise = probaNoiseReduction(arr)
    gapFilled = conicProbaPostProcessing(conicProbaPostProcessing(deNoise, 10, 0.35), 6, 0.35)
    zonesArr = probaToZones(gapFilled, zoneSize, probaThreshold)   
    noIslands = removeIslands(zonesArr, zoneSize*5, 1500, 5000, 18)
    noIslands = removeIslands(noIslands, zoneSize, 800, 3000, 18)
    noIslands = removeIslands(noIslands, 0, 400, 1600, 14)
    return noIslands


def removeIslands(arr, zoneSize, lowerIslandThreshold, upperIslandThreshold, ratioThreshold):
    newArr = arr.copy()
    examinedPoints = set()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 1 and (i, j) not in examinedPoints:
                island = getIslandArray(arr, (i, j), zoneSize)
                islandSize = len(island)
                if islandSize < upperIslandThreshold:
                    cluster_distance = find_max_distance(island)
                if upperIslandThreshold > islandSize > lowerIslandThreshold:
                    print("island size:", islandSize)
                    print("cluster distance:", cluster_distance)
                    print("ratio:", islandSize / cluster_distance)
                for k in range(islandSize):
                    examinedPoints.add(island[k])
                    if islandSize < upperIslandThreshold:
                        if islandSize < lowerIslandThreshold:
                            newArr[island[k][0]][island[k][1]] = 0
                        elif islandSize / cluster_distance > ratioThreshold:
                            newArr[island[k][0]][island[k][1]] = 0
    return newArr


def getIslandArray(arr, index, zoneSize):
    arrayOfPoints = []
    iMax = len(arr) - 1
    jMax = len(arr[0]) - 1
    i = index[0]
    j = index[1]
    FIFOQueue = deque([(i, j)])
    examinedElements = set()
    examinedElements.add((i, j))
    while (len(FIFOQueue) > 0):
        currentIndex = FIFOQueue.popleft()
        i = currentIndex[0]
        j = currentIndex[1]
        if i >= 0 and i < iMax and j >= 0 and j < jMax and arr[i][j] == 1:
            arrayOfPoints.append((i, j))
            #add horizontally and vertically
            if (i+1, j) not in examinedElements:
                FIFOQueue.append((i+1, j))
                examinedElements.add((i+1, j))
            if (i-1, j) not in examinedElements:
                FIFOQueue.append((i-1, j))
                examinedElements.add((i-1, j))
            if (i, j+1) not in examinedElements:
                FIFOQueue.append((i, j+1))
                examinedElements.add((i, j+1))
            if (i, j-1) not in examinedElements:
                FIFOQueue.append((i, j-1))
                examinedElements.add((i, j-1))
            #add diagonally
            if (i+1, j+1) not in examinedElements:
                FIFOQueue.append((i+1, j+1))
                examinedElements.add((i+1, j+1))
            if (i-1, j+1) not in examinedElements:
                FIFOQueue.append((i-1, j+1))
                examinedElements.add((i-1, j+1))
            if (i+1, j-1) not in examinedElements:
                FIFOQueue.append((i+1, j-1))
                examinedElements.add((i+1, j-1))
            if (i-1, j-1) not in examinedElements:
                FIFOQueue.append((i-1, j-1))
                examinedElements.add((i-1, j-1))
               
            #Add one zone away
            #add horizontally and vertically
            if (i+1 + zoneSize, j) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j))
                examinedElements.add((i+1 + zoneSize, j))
            if (i-1 - zoneSize, j) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j))
                examinedElements.add((i-1 - zoneSize, j))
            if (i, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i, j+1 + zoneSize))
                examinedElements.add((i, j+1 + zoneSize))
            if (i, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i, j-1 - zoneSize))
                examinedElements.add((i, j-1 - zoneSize))
            #add diagonally
            if (i+1 + zoneSize, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j+1 + zoneSize))
                examinedElements.add((i+1 + zoneSize, j+1 + zoneSize))
            if (i-1 - zoneSize, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j+1 + zoneSize))
                examinedElements.add((i-1 - zoneSize, j+1 + zoneSize))
            if (i+1 + zoneSize, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j-1 - zoneSize))
                examinedElements.add((i+1 + zoneSize, j-1 - zoneSize))
            if (i-1 - zoneSize, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j-1 - zoneSize))
                examinedElements.add((i-1 - zoneSize, j-1 - zoneSize))
    return arrayOfPoints

def conicProbaPostProcessing(arr, maskRadius, threshold):
    masks = []
    print("creating max mask")
    maxArr = gf(arr, np.nanmax, footprint=create_circular_mask(5))
    print("max mask created")
    for i in range(0, 8):
        masks.append(create_conic_mask(maskRadius, i))
    newArr = arr.copy()
    amountOfUpdated = 0
    examinedPoints = 0
    for i in range(len(arr)):
        print(i)
        for j in range(len(arr[i])):
            if arr[i][j] < 0.5 and maxArr[i][j] > 0.6:
                examinedPoints += 1
                trueProba = probaMeanFromMasks(arr, (i, j), masks)
                updatePixel = 0
                if trueProba[0] > threshold and trueProba[4] > threshold:
                    updatePixel = trueProba[0] if trueProba[0] > trueProba[4] else trueProba[4]
                if trueProba[1] > threshold and trueProba[5] > threshold:
                    updatePixelAgain = trueProba[1] if trueProba[1] > trueProba[5] else trueProba[5]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if trueProba[2] > threshold and trueProba[6] > threshold:
                    updatePixelAgain = trueProba[2] if trueProba[6] > trueProba[2] else trueProba[6]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if trueProba[3] > threshold and trueProba[7] > threshold:
                    updatePixelAgain = trueProba[3] if trueProba[3] > trueProba[7] else trueProba[7]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if updatePixel != 0:
                    amountOfUpdated += 1
                    if updatePixel < 0.5:
                        updatePixel *= 1.4
                    elif updatePixel < 0.55:
                        updatePixel *= 1.35
                    elif updatePixel < 0.6:
                        updatePixel *= 1.3
                    elif updatePixel < 0.65:
                        updatePixel *= 1.25
                    elif updatePixel < 0.7:
                        updatePixel *= 1.2
                    elif updatePixel < 0.75:
                        updatePixel *= 1.15
                    elif updatePixel < 0.85:
                        updatePixel *= 1.1
                    elif updatePixel < 0.9:
                        updatePixel *= 1.05
                    newArr[i][j] = updatePixel
    print(examinedPoints)
    print(amountOfUpdated)
    return newArr

def probaMeanFromMasks(arr, index, masks):
    row = index[0]
    col = index[1]
    halfMask = len(masks[0]) // 2    
    arrLenRow = len(arr)
    arrLenCol = len(arr[row])
    values = np.zeros(8)
    elementAmounts = np.zeros(8)
    for i in range(-halfMask , halfMask):
        for j in range(-halfMask , halfMask):
            if arrLenCol > col + j + 1 and col + j + 1 >= 0 and arrLenRow > row + i + 1 and row + i + 1 >= 0:
                if masks[0][i + halfMask][j + halfMask] == 1:
                    values[0] += arr[row + i][col + j]
                    elementAmounts[0] += 1
                elif masks[1][i + halfMask][j + halfMask] == 1:
                    values[1] += arr[row + i][col + j]
                    elementAmounts[1] += 1
                elif masks[2][i + halfMask][j + halfMask] == 1:
                    values[2] += arr[row + i][col + j]
                    elementAmounts[2] += 1
                elif masks[3][i + halfMask][j + halfMask] == 1:
                    values[3] += arr[row + i][col + j]
                    elementAmounts[3] += 1
                elif masks[4][i + halfMask][j + halfMask] == 1:
                    values[4] += arr[row + i][col + j]
                    elementAmounts[4] += 1
                elif masks[5][i + halfMask][j + halfMask] == 1:
                    values[5] += arr[row + i][col + j]
                    elementAmounts[5] += 1
                elif masks[6][i + halfMask][j + halfMask] == 1:
                    values[6] += arr[row + i][col + j]
                    elementAmounts[6] += 1
                elif masks[7][i + halfMask][j + halfMask] == 1:
                    values[7] += arr[row + i][col + j]
                    elementAmounts[7] += 1
    for i in range(len(values)):
        values[i] = values[i] / elementAmounts[i] if elementAmounts[i] != 0 else 0
    return values

"""------------------------------------------------------------------------------------------------------------------------------"""