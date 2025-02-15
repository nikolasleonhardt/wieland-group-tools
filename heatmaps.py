#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
from scipy.stats import gaussian_kde
import os
import pandas as pd
import math as m

number_of_bins:int = 50
number_of_measurement_rows:int = 5
frame_time_row:int = 0
radius_row:int = 1
angle_row:int = 2
signal_1_row:int = 3
signal_2_row:int = 4
outputappendix:str = 'centerEdgeRatios'
inputappendix:str = ''
inputfolder = 'allNSF/'
mouse_id_length:int = 2
sigma_deviation_cutoff:int = 3
sensor = 'one'
number_of_2d_bins:int = 100
path_of_script = os.path.dirname(os.path.realpath(__file__))
outputpath = os.path.join(path_of_script, outputappendix)
z_max = 0
zAreaMax = 0
colorPlotMax = 0.0036

def extract_mouse_id(file_name, mouse_id_length):
    mouse_id = ''
    count = 0
    for char in file_name:
        if char.isdigit():
            mouse_id += char
            count += 1
            if count == mouse_id_length:
                break
    return mouse_id

def createSingleLocationHeatmap(filepath, weighted=False, savePlot=False, centerEdgeRatioBool=False):
    global z_max
    global zAreaMax
    print(filepath)
    norm = colors.Normalize(vmin=0., vmax=colorPlotMax)
    mouseID = str(extract_mouse_id(filepath, mouse_id_length))
    current_data_dataframe = pd.read_csv(filepath, sep=',', header=None)
    current_data_array = current_data_dataframe.to_numpy()
    radius_data_array = current_data_array[:,radius_row]
    phi_data_array = current_data_array[:,angle_row]
    photometry_array_1 = current_data_array[:,signal_1_row]
    x_array = radius_data_array*np.cos(phi_data_array)
    y_array = radius_data_array*np.sin(phi_data_array)

    low_calcium_markers = np.ma.masked_greater_equal(photometry_array_1, sigma_deviation_cutoff)
    high_calcium_markers = np.ma.masked_less(photometry_array_1, sigma_deviation_cutoff)
    high_mask = high_calcium_markers.mask
    highX = x_array[~high_mask]
    highY = y_array[~high_mask]
    lengthOfExperiment = np.size(radius_data_array)

    if weighted and highX.size == 0:
        print('Dieses Tier hatte keine Messungen über dem Schwellenwert!')
        return x_array, y_array, highX, highY, np.nan

    
    xyHigh = np.vstack([highX, highY])
    xy = np.vstack([x_array, y_array])
    kdeHigh = gaussian_kde(xyHigh)
    kde = gaussian_kde(xy)

    xgrid, ygrid = np.mgrid[-1:1:100j, -1:1:100j]
    propabilityLocation = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    propabilityLocationHigh = kdeHigh(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    probabilityHigh = highX.size/x_array.size
    #zgridBayesian = propabilityLocationHigh * probabilityHigh / propabilityLocation
    zgridBayesian = propabilityLocationHigh 
    #plt.contourf(xgrid, ygrid, zgridNormalized, 50, cmap='viridis', vmin=0, vmax=1)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis', levels=np.linspace(0., colorPlotMax, 100), norm=norm)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis')
    if weighted:
        plt.contourf(xgrid, ygrid, zgridBayesian, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, propabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        outputpath = os.path.join(path_of_script, outputappendix, filepath.split('/')[-2])
        plt.savefig(os.path.join(outputpath, mouseID+'heatmap.png'))
    plt.clf()
    if centerEdgeRatioBool:
        centerEdgeRatio = getCenterEdgeRatios(zgridBayesian, 0.2)
        return x_array, y_array, highX, highY, centerEdgeRatio
    return x_array, y_array, highX, highY

def createBatchLocationHeatmaps(folderpath, weighted=False, savePlot=False, centerEdgeRatioBool=False, jackknifeBool=False, jackknifeFile=None):
    global z_max
    norm = colors.Normalize(vmin=0., vmax=colorPlotMax)
    print(folderpath)
    transformed_data_files = [f for f in os.listdir(folderpath) if not f.startswith('.')]
    if jackknifeBool:
        transformed_data_files.remove(jackknifeFile)
    x_combined, y_combined, highX_combined, highY_combined, centerEdgeRatioArray = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    miceList = []
    for index, filename in enumerate(transformed_data_files):
        current_path = os.path.join(folderpath, filename)
        miceList.append(filename)
        if centerEdgeRatioBool:
            x_buffer, y_buffer, highX_buffer, highY_buffer, mouseCenterEdgeRatio = createSingleLocationHeatmap(current_path, weighted=weighted, savePlot=savePlot, centerEdgeRatioBool=centerEdgeRatioBool)
            centerEdgeRatioArray = np.append(centerEdgeRatioArray, np.array([mouseCenterEdgeRatio]))
        else:
            x_buffer, y_buffer, highX_buffer, highY_buffer = createSingleLocationHeatmap(current_path, weighted=weighted, savePlot=savePlot, centerEdgeRatioBool=centerEdgeRatioBool)
        x_combined = np.concatenate([x_combined, x_buffer])
        y_combined = np.concatenate([y_combined, y_buffer])
        highX_combined = np.concatenate([highX_combined, highX_buffer])
        highY_combined = np.concatenate([highY_combined, highY_buffer])
    xyLocationCombined = np.vstack([x_combined, y_combined])
    xyHighCombined = np.vstack([highX_combined, highY_combined])
    kdeLocation = gaussian_kde(xyLocationCombined)
    kdeHigh = gaussian_kde(xyHighCombined)
    xgrid, ygrid = np.mgrid[-1:1:100j, -1:1:100j]
    probabilityLocation = kdeLocation(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    probabilityLocationHigh = kdeHigh(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    probabilityHigh = highX_combined.size/x_combined.size
    #zGridBayesian = probabilityLocationHigh * probabilityHigh / probabilityLocation
    zGridBayesian = probabilityLocationHigh
    #plt.contourf(xgrid, ygrid, zgridNormalized, 50, cmap='viridis', vmin=0, vmax=1)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis', levels=np.linspace(0., colorPlotMax, 100), norm=norm)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis')
    if centerEdgeRatioBool:
        finalArray = np.vstack([miceList, centerEdgeRatioArray])
        np.savetxt(os.path.join(path_of_script, outputappendix, 'centerEdgeRatios.csv'), finalArray.T, delimiter=',', fmt='%s')
        cleanCenterEdgeRatioArray = centerEdgeRatioArray[~np.isnan(centerEdgeRatioArray)]
        cleanCenterEdgeRatioArray = cleanCenterEdgeRatioArray[~np.isinf(cleanCenterEdgeRatioArray)]
        centerEdgeRatioMean = np.mean(cleanCenterEdgeRatioArray)
        centerEdgeRatioSem = np.std(cleanCenterEdgeRatioArray)/np.sqrt(np.size(cleanCenterEdgeRatioArray))
        meanFinalArray = np.array([centerEdgeRatioMean, centerEdgeRatioSem])
        np.savetxt(os.path.join(path_of_script, outputappendix, 'centerEdgeRatiosMean.csv'), meanFinalArray, delimiter=',', fmt='%s', header='Mean, SEM')
    if weighted:
        plt.contourf(xgrid, ygrid, zGridBayesian, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, probabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        outputpath = os.path.join(path_of_script, outputappendix, folderpath)
        plt.savefig(os.path.join(outputpath, 'combined_heatmap.png'))
    plt.clf()
    if jackknifeBool:
        centerEdgeJackknife = getCenterEdgeRatios(zGridBayesian, 0.2)
        return x_combined, y_combined, highX_combined, highY_combined, centerEdgeJackknife
    return x_combined, y_combined, highX_combined, highY_combined

def createAllMiceLocationHeatmap(folderPathList, weighted=False, savePlot=False):
    global z_max
    norm = colors.Normalize(vmin=0, vmax=colorPlotMax)
    x_combined, y_combined, highX_combined, highY_combined = np.array([]), np.array([]), np.array([]), np.array([])
    for folder in folderPathList:
        print(folder)
        x_buffer, y_buffer, highX_buffer, highY_buffer = createBatchLocationHeatmaps(folder, weighted=weighted, savePlot=savePlot)
        x_combined = np.concatenate([x_combined, x_buffer])
        y_combined = np.concatenate([y_combined, y_buffer])
        highX_combined = np.concatenate([highX_combined, highX_buffer])
        highY_combined = np.concatenate([highY_combined, highY_buffer])
    xyLocationCombined = np.vstack([x_combined, y_combined])
    xyHighCombined = np.vstack([highX_combined, highY_combined])
    kdeLocation = gaussian_kde(xyLocationCombined)
    kdeHigh = gaussian_kde(xyHighCombined)
    xgrid, ygrid = np.mgrid[-1:1:100j, -1:1:100j]
    probabilityLocation = kdeLocation(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    probabilityLocationHigh = kdeHigh(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    probabilityHigh = highX_combined.size/x_combined.size
    #zgridBayesian = probabilityLocationHigh * probabilityHigh / probabilityLocation
    zgridBayesian = probabilityLocationHigh
    #plt.contourf(xgrid, ygrid, zgridNormalized, 50, cmap='viridis', vmin=0, vmax=1)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis', levels=np.linspace(0., colorPlotMax, 100), norm=norm)
    #plt.contourf(xgrid, ygrid, zgridAreaNormalized, 50, cmap='viridis')
    if weighted:
        plt.contourf(xgrid, ygrid, zgridBayesian, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, probabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        plt.savefig(os.path.join(path_of_script, outputappendix, 'allMiceheatmap.png'))
    plt.clf()
    return zgridBayesian

def calculateBhattacharyya(distribution1, distribution2):
    distribution1 = distribution1/np.size(distribution1)
    distribution2 = distribution2/np.size(distribution2)
    coefficient = np.sum(np.sqrt(distribution1*distribution2))
    distance = -m.log(coefficient)
    return distance, coefficient

def getCenterEdgeRatios(distribution, percentageOfEdgeLength):
    originalEdgeLength = np.size(distribution[0])
    edgeLength = int(percentageOfEdgeLength*originalEdgeLength)
    lowerBound = int((originalEdgeLength - edgeLength)/2)
    upperBound = int((originalEdgeLength + edgeLength)/2)
    center = distribution[lowerBound:upperBound, lowerBound:upperBound]
    edge = np.concatenate([distribution[0:lowerBound, 0:originalEdgeLength].flatten(), distribution[0:originalEdgeLength, 0:lowerBound].flatten(), distribution[0:originalEdgeLength, upperBound:originalEdgeLength].flatten(), distribution[upperBound:originalEdgeLength, 0:originalEdgeLength].flatten()])
    centerSum = np.mean(center)
    edgeSum = np.mean(edge)
    return centerSum/edgeSum

def jackknife(folderPath, weighted=False, savePlot=False):
    filesToAnalyze = [f for f in os.listdir(folderPath) if not f.startswith('.')]
    jackknifeArray = np.array([])
    jackknifedFileList = []
    for index, file in enumerate(filesToAnalyze):
        print('Jackknife: ' + file)
        jackknifedFileList.append(file)
        x_buffer, y_buffer, highX_buffer, highY_buffer, centerEdgeRatio = createBatchLocationHeatmaps(folderPath, weighted=weighted, savePlot=savePlot, centerEdgeRatioBool=True, jackknifeBool=True, jackknifeFile=file)
        jackknifeArray = np.append(jackknifeArray, centerEdgeRatio)
    jackknifedFileArray = np.array(jackknifedFileList)
    combinedFinalArray = np.vstack([jackknifedFileArray, jackknifeArray])
    np.savetxt(os.path.join(path_of_script, outputappendix, 'jackknifeCenterEdgeRatios.csv'), combinedFinalArray.T, delimiter=',', fmt='%s')
    heatmapsMean = np.mean(jackknifeArray)
    heatmapsSem = np.std(jackknifeArray)/np.sqrt(np.size(jackknifeArray))
    print(heatmapsMean, heatmapsSem)

os.makedirs(outputpath, exist_ok=True)
jackknife(os.path.join(path_of_script, inputfolder), weighted=True, savePlot=False)