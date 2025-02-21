#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
from scipy.stats import gaussian_kde
import os
import pandas as pd
import math as m
import warnings 

#Set constant variables for the analysis
number_of_bins:int = 50
number_of_measurement_rows:int = 5
frame_time_row:int = 0
radius_row:int = 1
angle_row:int = 2
signal_1_row:int = 3
signal_2_row:int = 4
sigma_deviation_cutoff:int = 3
sensor = 'one'
number_of_2d_bins:int = 100

#Set constant variables for the file structure
outputappendix:str = 'heatmaps'
inputappendix:str = ''
inputfolder:str = 'transformedData'
mouse_id_length:int = 2

path_of_script = os.path.dirname(os.path.realpath(__file__))
outputpath = os.path.join(path_of_script, outputappendix)
os.makedirs(outputpath, exist_ok=True)

z_max = 0
zAreaMax = 0
colorPlotMax = 0.0036

def extract_mouse_id(file_name: str, mouse_id_length: int) -> str:
    """
    Extracts the mouse id from the given file name.

    Parameters:
        file_name (str): The name of the file from which to extract the mouse id.
        mouse_id_length (int): The length of the mouse id.

    Returns:
        str: The extracted mouse id.
    """
    mouse_id = ''
    count = 0
    for char in file_name:
        if char.isdigit():
            mouse_id += char
            count += 1
            if count == mouse_id_length:
                break
    return mouse_id

def createSingleLocationHeatmap(filepath: str, weighted: bool=False, savePlot: bool=False, centerEdgeRatioBool: bool=False):
    """
    Creates a heatmap of the location of the mouse during the experiment. The heatmap is created using a gaussian kernel density estimation 
    and represents a probability estimate of the mouse being in a certain location.

    Parameters:
        filepath (str): The path to the file containing the data.
        weighted (bool): A boolean value indicating whether the heatmap should be calculated using only calcium measurements above a given sigma treshold.
        savePlot (bool): A boolean value indicating whether the plot should be saved.
        centerEdgeRatioBool (bool): A boolean value indicating whether the center-edge ratio should be calculated for the caclulated heatmap should be calculated.
    
    Returns:
        tuple: A tuple containing the x and y coordinates of the mouse, the x and y coordinates of the mouse when the signal strength is above the threshold
                and (if calculated) the center-edge ratio. 
    """
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

    if weighted:
        xyHigh = np.vstack([highX, highY])
        try:
            kdeHigh = gaussian_kde(xyHigh)
        except:
            warnings.warn('This mouse did not have sufficient high calcium measurements. Therefore no heatmap will be created.')
            return x_array, y_array, highX, highY

    xy = np.vstack([x_array, y_array])
    kde = gaussian_kde(xy)

    xgrid, ygrid = np.mgrid[-1:1:100j, -1:1:100j]
    probabilityLocation = kde(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    if weighted:
        probabilityLocationHigh = kdeHigh(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
        plt.contourf(xgrid, ygrid, probabilityLocationHigh, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, probabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        outputpath = os.path.join(path_of_script, outputappendix)
        plt.savefig(os.path.join(outputpath, mouseID+'heatmap.png'))
    plt.clf()
    if centerEdgeRatioBool:
        centerEdgeRatio = getCenterEdgeRatios(probabilityLocationHigh, 0.2)
        return x_array, y_array, highX, highY, centerEdgeRatio
    return x_array, y_array, highX, highY

def createBatchLocationHeatmaps(folderpath: str, weighted: bool=False, savePlot: bool=False, centerEdgeRatioBool: bool=False, jackknifeBool: bool=False, jackknifeFile: bool=None):
    """
    Creates both the individual heatmpas for all mice in a given folder and a combined heatmap of all the measurements in the folder, obtained by concatenating
    the individual measurements and performing a gaussian kernel density estimation on the combined data.
    Additionally the center-edge ratio can be calculated for each mouse and the combined data.

    Parameters:
        folderpath (str): The path to the folder containing the data.
        weighted (bool): A boolean value indicating whether the heatmap should be calculated using only calcium measurements above a given sigma treshold.
        savePlot (bool): A boolean value indicating whether the resulting plots should be saved.
        centerEdgeRatioBool (bool): A boolean value indicating whether the center-edge ratio should be calculated for the caclulated heatmap should be calculated.
        jackknifeBool (bool): A boolean value indicating whether for use in jackknife resampling an experiment run should be omitted from the combined heatmap
                                and the center-edge ratio calculation.
        jackknifeFile (str): The name of the file to be omitted from the combined heatmap and center-edge ratio calculation.
    """
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
    xgrid, ygrid = np.mgrid[-1:1:100j, -1:1:100j]
    xyLocationCombined = np.vstack([x_combined, y_combined])
    kdeLocation = gaussian_kde(xyLocationCombined)
    probabilityLocation = kdeLocation(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
    if weighted:
        xyHighCombined = np.vstack([highX_combined, highY_combined])
        try:
            kdeHigh = gaussian_kde(xyHighCombined)
        except:
            warnings.warn('There were not enough high calcium measurements to create a heatmap.')
            return x_combined, y_combined, highX_combined, highY_combined
        probabilityLocationHigh = kdeHigh(np.vstack([xgrid.ravel(), ygrid.ravel()])).reshape(xgrid.shape)
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
        plt.contourf(xgrid, ygrid, probabilityLocationHigh, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, probabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        outputpath = os.path.join(path_of_script, outputappendix)
        plt.savefig(os.path.join(outputpath, 'combined_heatmap.png'))
    plt.clf()
    if jackknifeBool:
        centerEdgeJackknife = getCenterEdgeRatios(probabilityLocationHigh, 0.2)
        return x_combined, y_combined, highX_combined, highY_combined, centerEdgeJackknife
    return x_combined, y_combined, highX_combined, highY_combined

def createAllMiceLocationHeatmap(folderPathList: list, weighted: bool=False, savePlot: bool=False):
    """
    Works the same as createBatchLocationHeatmaps, but is called on a list of folders instead of a single folder. For each folder 'createBatchLocationHeatmaps'
    is called. In the end a combined heatmap of all the data is created.

    Parameters:
        folderPathList (list): A list of the paths to the folders containing the data.
        weighted (bool): A boolean value indicating whether the heatmap should be calculated using only calcium measurements above a given sigma treshold.
        savePlot (bool): A boolean value indicating whether the resulting plots should be saved.

    Returns:
        None
    """
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
    if weighted:
        plt.contourf(xgrid, ygrid, probabilityLocationHigh, 50, cmap='viridis')
    else:
        plt.contourf(xgrid, ygrid, probabilityLocation, 50, cmap='viridis')
    plt.colorbar(label='probability density')
    if savePlot:
        plt.savefig(os.path.join(path_of_script, outputappendix, 'allMiceheatmap.png'))
    plt.clf()
    return None

def calculateBhattacharyya(distribution1: np.ndarray, distribution2: np.ndarray):
    """
    Calculates the Bhattacharyya distance and coefficient between two given distributions.

    Parameters:
        distribution1 (np.ndarray): The first distribution.
        distribution2 (np.ndarray): The second distribution.
    
    Returns:
        tuple: A tuple containing the Bhattacharyya distance and coefficient.
    """
    distribution1 = distribution1/np.size(distribution1)
    distribution2 = distribution2/np.size(distribution2)
    coefficient = np.sum(np.sqrt(distribution1*distribution2))
    distance = -m.log(coefficient)
    return distance, coefficient

def getCenterEdgeRatios(distribution: np.ndarray, percentageOfEdgeLength: float):
    """
    Calculates the center-edge ratio of a given distribution. The center-edge ratio is defined as the ratio of the mean of the center of the distribution 
    to the mean of the edge of the distribution, where the edge is defined as the outermost 'percentageOfEdgeLength' of the distribution and the center as the
    innermost 'percentageOfEdgeLength' of the distribution.

    Parameters:
        distribution (np.ndarray): The distribution for which to calculate the center-edge ratio.
        percentageOfEdgeLength (float): The percentage of the edge length to use for the calculation of center and edge.
    
    Returns:
        centerEdgeRatio (float): The calculated center-edge ratio.
    """
    originalEdgeLength = np.size(distribution[0])
    edgeLength = int(percentageOfEdgeLength*originalEdgeLength)
    lowerBound = int((originalEdgeLength - edgeLength)/2)
    upperBound = int((originalEdgeLength + edgeLength)/2)
    center = distribution[lowerBound:upperBound, lowerBound:upperBound]
    edge = np.concatenate([distribution[0:lowerBound, 0:originalEdgeLength].flatten(), distribution[0:originalEdgeLength, 0:lowerBound].flatten(), distribution[0:originalEdgeLength, upperBound:originalEdgeLength].flatten(), distribution[upperBound:originalEdgeLength, 0:originalEdgeLength].flatten()])
    centerSum = np.mean(center)
    edgeSum = np.mean(edge)
    centerEdgeRatio = centerSum/edgeSum
    return centerEdgeRatio

def jackknife(folderPath: str, weighted: bool=False, savePlot: bool=False):
    """
    Uses the jackknife resampling method to estimate the center-edge ratio and its error for the data in the given folder.
    The values obtained by dropping a specific file are saved in a csv file called 'jackknifeCenterEdgeRatios.csv'.
    The final result for the mean and standard error of the center-edge ratio are saved in a file called 'jackknifeCenterEdgeRatiosMean.csv'.
    Depending on the number of files included in the analysis, this function can take a long time to run!

    Parameters:
        folderPath (str): The path to the folder containing the data.
        weighted (bool): A boolean value indicating whether the heatmap should be calculated using only calcium measurements above a given sigma treshold.
        savePlot (bool): A boolean value indicating whether the resulting plots should be saved.
    
    Returns:
        None
    """
    filesToAnalyze = [f for f in os.listdir(folderPath) if not f.startswith('.')]
    jackknifeArray = np.array([])
    jackknifedFileList = []
    for index, file in enumerate(filesToAnalyze):
        print('Jackknifed: ' + file)
        jackknifedFileList.append(file)
        x_buffer, y_buffer, highX_buffer, highY_buffer, centerEdgeRatio = createBatchLocationHeatmaps(folderPath, weighted=weighted, savePlot=savePlot, centerEdgeRatioBool=True, jackknifeBool=True, jackknifeFile=file)
        jackknifeArray = np.append(jackknifeArray, centerEdgeRatio)
    jackknifedFileArray = np.array(jackknifedFileList)
    combinedFinalArray = np.vstack([jackknifedFileArray, jackknifeArray])
    np.savetxt(os.path.join(path_of_script, outputappendix, 'jackknifeCenterEdgeRatios.csv'), combinedFinalArray.T, delimiter=',', fmt='%s')
    heatmapsMean = np.mean(jackknifeArray)
    heatmapsSem = np.std(jackknifeArray)/np.sqrt(np.size(jackknifeArray))
    np.savetxt(os.path.join(path_of_script, outputappendix, 'jackknifeCenterEdgeRatiosMean.csv'), np.array([heatmapsMean, heatmapsSem]), delimiter=',', fmt='%s')
    return None

#Script starts here:
createSingleLocationHeatmap(os.path.join(path_of_script, inputfolder, '2.txt'), weighted=True, savePlot=True)
createBatchLocationHeatmaps(os.path.join(path_of_script, inputfolder), weighted=True, savePlot=True)