# General arr lib
import numpy as np

# Compile py code
from numba import jit
from numba import prange

# Multithreading
import dask.array as da
import dask as dk
dk.config.set(scheduler='processes')

from dask_image.ndfilters import generic_filter as d_gf

from collections import deque

from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from skimage.filters import gabor
from skimage.restoration import  denoise_bilateral

import datetime

@jit
def create_circular_mask(radius):
    """
    Creates a circular mask
    """
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    mask[radius][radius] = 0
    kernel[mask] = 1
    return kernel


@jit(nopython=True)
def _reclassify_impoundment(arr):
    """
    Internaly used normalization of impoundment index reclassification different threashhold
    """
    new_arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if new_arr[i, j] == 0:
                new_arr[i, j] = 0
            elif new_arr[i, j] < 0.002:
                new_arr[i, j] = 5
            elif arr[i, j] < 0.005:
                new_arr[i, j] = 50
            elif arr[i, j] < 0.02:
                new_arr[i, j] = 100
            elif arr[i, j] < 0.05:
                new_arr[i, j] = 1000
            elif arr[i, j] < 0.1:
                new_arr[i, j] = 10000
            elif arr[i, j] < 0.3:
                new_arr[i, j] = 100000
            else:
                new_arr[i, j] = 1000000
    return new_arr


@jit
def impoundmentAmplification(arr, mask_radius=10):
    """
    Amplicatates ditches
    """
    norm_arr = da.from_array(_reclassify_impoundment(arr), chunks=(800, 800))
    mask = create_circular_mask(mask_radius)
    return d_gf(d_gf(d_gf(norm_arr, np.nanmean, footprint=mask), np.nanmean, footprint=mask), np.nanmedian, footprint=mask).compute(scheduler='processes')


@jit(nopython=True)
def _reclassify_hpmf_filter(arr):
    """
    Internal reclassification wrapper
    """
    binary = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.000001 and arr[i][j] > -0.000001:
                binary[i][j] = 100
            else:
                binary[i][j] = 0
    return binary


@jit(nopython=True)
def _reclassify_hpmf_filter_mean(arr):
    """
    Internal reclassification wrapper
    """
    reclassify = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 1:
                reclassify[i][j] = 0
            elif arr[i][j] < 3:
                reclassify[i][j] = 1
            elif arr[i][j] < 7:
                reclassify[i][j] = 2
            elif arr[i][j] < 10:
                reclassify[i][j] = 50
            elif arr[i][j] < 20:
                reclassify[i][j] = 75
            elif arr[i][j] < 50:
                reclassify[i][j] = 100
            elif arr[i][j] < 80:
                reclassify[i][j] = 300
            elif arr[i][j] < 100:
                reclassify[i][j] = 600
            else:
                reclassify[i][j] = 1000
    return reclassify


@jit
def hpmfFilter(arr):
    """
    HPMF filter enchances ditches
    """
    normalized_arr = da.from_array(
        _reclassify_hpmf_filter(arr), chunks=(800, 800))

    mean = d_gf(d_gf(d_gf(d_gf(normalized_arr, np.amax, footprint=create_circular_mask(1)), np.amax, footprint=create_circular_mask(
        1)), np.median, footprint=create_circular_mask(2)), np.nanmean, footprint=create_circular_mask(5)).compute(scheduler='processes')
    reclassify = da.from_array(
        _reclassify_hpmf_filter_mean(mean), chunks=(800, 800))

    return d_gf(reclassify, np.nanmean, footprint=create_circular_mask(7))


@jit(nopython=True)
def _reclassify_sky_view_non_ditch_amp(arr):
    """
    Internal amp reclassification
    """
    new_arr = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i, j] < 0.92:
                new_arr[i, j] = 46
            elif arr[i, j] < 0.93:
                new_arr[i, j] = 37
            elif arr[i, j] < 0.94:
                new_arr[i, j] = 29
            elif arr[i, j] < 0.95:
                new_arr[i, j] = 22
            elif arr[i, j] < 0.96:
                new_arr[i, j] = 16
            elif arr[i, j] < 0.97:
                new_arr[i, j] = 11
            elif arr[i, j] < 0.98:
                new_arr[i, j] = 7
            elif arr[i, j] < 0.985:
                new_arr[i, j] = 4
            elif arr[i, j] < 0.99:
                new_arr[i, j] = 2
            else:
                new_arr[i, j] = 1
    return new_arr


@jit
def _skyViewGabor(merged, gabors):
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            merged[i][j] = 0
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            for k in range(len(gabors)):
                merged[i][j] += gabors[k][i][j]
    return merged

#@jit
def skyViewGabor(skyViewArr):
    delayed_gabors = []
    for i in np.arange(0.03, 0.08, 0.01):
        for j in np.arange(0, 3, 0.52):
            delayed_gabor = dk.delayed(gabor)(skyViewArr, i, j)[0]
            delayed_gabors.append(delayed_gabor)
    gabors = dk.compute(delayed_gabors)
    print(type(gabors[0]))
    print(len(gabors[0]))
    #print(len(gabors))
    #gabors = gabors.map(lambda x : x[0])
    return _skyViewGabor(skyViewArr.copy(), gabors[0])





@jit
def skyViewNonDitchAmplification(arr):
    arr = da.from_array(arr, chunks=(800, 800))
    arr = d_gf(arr, np.nanmedian, footprint=create_circular_mask(25)
               ).compute(scheduler='processes')
    arr = da.from_array(
        _reclassify_sky_view_non_ditch_amp(arr), chunks=(800, 800))
    return d_gf(arr, np.nanmean, footprint=create_circular_mask(10))


@jit
def conicMean(arr, maskRadius, threshold):
    # Standard values: maskRadius = 5, threshold = 0.975
    masks = []
    for i in range(0, 8):
        masks.append(create_conic_mask(maskRadius, i))
    newArr = arr.copy()
    amountOfThresholds = 0
    for i in range(len(arr)):
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
                newArr[i][j] = 0.95 * lowest if lowest * \
                    0.95 < arr[i][j] else arr[i][j]
    print(amountOfThresholds)
    return newArr


@jit
def meanFromMasks(arr, index, masks):
    row = index[0]
    col = index[1]
    halfMask = len(masks[0]) // 2
    arrLenRow = len(arr)
    arrLenCol = len(arr[row])
    values = np.zeros(8)
    elementAmounts = np.zeros(8)
    for i in range(-halfMask, halfMask):
        for j in range(-halfMask, halfMask):
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
        values[i] = values[i] / \
            elementAmounts[i] if elementAmounts[i] != 0 else 0.99
    return values


@jit
def create_conic_mask(radius, direction):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    if direction == 0:  # topright
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 1:  # righttop
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 2:  # rightbottom
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 3:  # bottomright
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 4:  # bottomleft
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x < 0)
    elif direction == 5:  # leftbottom
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (
            x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 6:  # lefttop
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (
            x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 7:  # topleft
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x < 0)
    kernel[mask] = 1
    return kernel


@jit(nopython=True)
def _slopeNonDitchAmplifcation_normalize(arr, new_arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 8:
                new_arr[i][j] = 0
            elif arr[i][j] < 9:
                new_arr[i][j] = 20
            elif arr[i][j] < 10:
                new_arr[i][j] = 25
            elif arr[i][j] < 11:
                new_arr[i][j] = 30
            elif arr[i][j] < 13:
                new_arr[i][j] = 34
            elif arr[i][j] < 15:
                new_arr[i][j] = 38
            elif arr[i][j] < 17:
                new_arr[i][j] = 42
            elif arr[i][j] < 19:
                new_arr[i][j] = 46
            elif arr[i][j] < 21:
                new_arr[i][j] = 50
            else:
                new_arr[i][j] = 55
    return new_arr


@jit
def slopeNonDitchAmplification(arr):
    new_arr = arr.copy()
    arr = d_gf(da.from_array(arr, chunks=(800, 800)), np.nanmedian,
               footprint=create_circular_mask(35)).compute(scheduler='processes')
    new_arr = _slopeNonDitchAmplifcation_normalize(arr, new_arr)
    return d_gf(da.from_array(new_arr, chunks=(800, 800)), np.nanmean, footprint=create_circular_mask(15))


@jit(nopython=True)
def rasterToZones(arr, zoneSize, threshold):
    newArr = arr.copy()
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


@jit(nopython=True)
def probaToZones(arr, zoneSize, threshold):
    newArr = np.zeros(arr.shape)
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


@jit(nopython=True)
def _customeRemoveNoise(arr, max_arr, new_arr, threshold, selfThreshold):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if max_arr[i][j] < threshold:
                if arr[i][j] > selfThreshold:
                    new_arr[i][j] *= 0.5
                else:
                    new_arr[i][j] *= 0.25
    return new_arr


@jit
def customRemoveNoise(arr, radius, threshold, selfThreshold):
    max_arr = d_gf(da.from_array(arr, chunks=(800, 800)), np.nanmax,
                   footprint=create_circular_mask(radius)).compute(scheduler='processes')
    return _customeRemoveNoise(arr, max_arr, np.copy(arr), threshold, selfThreshold)

def find_max_distance(A):
    """
    Returns the maximum distance from  2x points
    each point is represented by a x,y cord.
    """
    #assert(A.shape[1] == 2)
    return nanmax(squareform(pdist(A)))


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
                for k in range(islandSize):
                    examinedPoints.add(island[k])
                    if islandSize < upperIslandThreshold:
                        if islandSize < lowerIslandThreshold:
                            newArr[island[k][0]][island[k][1]] = 0
                        elif islandSize / cluster_distance > ratioThreshold:
                            newArr[island[k][0]][island[k][1]] = 0
    return newArr      
        
@jit
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
            # add horizontally and vertically
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
            # add diagonally
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

            # Add one zone away
            # add horizontally and vertically
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
            # add diagonally
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



# ----------------------- post -----------------

#@jit("float64[:](float64[:,:], int32, int32, int32[:, :, :] )", nopython=True)
@jit(nopython=True)
def probaMeanFromMasks(arr, row, col, masks):
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


#@jit("float64[:,:](float64[:,:], float64[:,:], int32[:,:,:], float64)", nopython=True)
@jit(nopython=True)
def _conicProbaPostProcessing(arr, maxArr, masks, threshold):
    newArr = arr.copy()
    amountOfUpdated = 0
    examinedPoints = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.5 and maxArr[i][j] > 0.6:
                examinedPoints += 1
                trueProba = probaMeanFromMasks(arr, i, j, masks)
                
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
    return newArr

@jit
def conicProbaPostProcessing(arr, maskRadius, threshold):
    masks = []
    maxArr = d_gf(da.from_array(arr,chunks = (800,800)), np.nanmax, footprint=create_circular_mask(5))
    for i in range(0, 8):
        masks.append(create_conic_mask(maskRadius, i))

    return _conicProbaPostProcessing(np.array(arr), np.array(maxArr), np.array(masks),threshold)
    
def __denoise_bilateral(arr):
    return denoise_bilateral(arr, sigma_spatial=15, multichannel=False)

#@jit("float64[:,:](float64[:,:])")
def probaNoiseReduction(arr):
    d = da.from_array(arr, chunks=(800,800))
    return customRemoveNoise(d.map_overlap(__denoise_bilateral, depth=15).compute(), 10, 0.7, 0.4)
    

#@jit("float64[:,:](float64[:,:], int32, float64)")
def probaPostProcess(arr, zoneSize, probaThreshold):
    print("started:", str(datetime.datetime.now().hour), str(datetime.datetime.now().minute) )
    deNoise = probaNoiseReduction(arr)
    print("deNoise done:", str(datetime.datetime.now().hour), str(datetime.datetime.now().minute) )
    gapFilled = conicProbaPostProcessing(conicProbaPostProcessing(deNoise, 8, 0.35), 5, 0.3)
    print("gapFill done:", str(datetime.datetime.now().hour), str(datetime.datetime.now().minute) )
    zonesArr = probaToZones(gapFilled, zoneSize, 0.4)
    print("probaToZone done:", str(datetime.datetime.now().hour), str(datetime.datetime.now().minute) )
    noIslands = removeIslands(zonesArr, zoneSize*2, 1500, 10000, 30)
    noIslands = removeIslands(noIslands, zoneSize, 1000, 5000, 20)
    noIslands = removeIslands(noIslands, 0, 500, 3000, 18)
    noIslands = removeIslands(noIslands, 0, 500, 1200, 14)
    return noIslands