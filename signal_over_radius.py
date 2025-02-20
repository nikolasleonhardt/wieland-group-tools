import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from scipy.optimize import curve_fit

#Constants for the analysis:
path_of_script = os.path.dirname(os.path.realpath(__file__))
number_of_measurement_rows:int = 5
frame_time_row:int = 0
radius_row:int = 1
angle_row:int = 2
signal_1_row:int = 3
signal_2_row:int = 4
list_of_wanted_bin_numbers = [10]
fit = True
output_dir:str = os.path.join(path_of_script, 'signal_over_radius')
mouse_id_length:int = 2
os.makedirs(output_dir, exist_ok=True)

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

def getRadiusAverage(input_array: np.ndarray) -> np.ndarray:
    """
    Takes an array of radii and returns an array of the average radius between each pair of radii.

    Parameters:
        input_array (np.ndarray): The array of radii.
    Returns:    
        np.ndarray: The array of the average radius between each pair of radii.
    """
    output_array = np.zeros(np.size(input_array)-1)
    for i in range(np.size(input_array)-1):
        output_array[i] = (input_array[i]+input_array[i+1])/2
    return output_array

def linearModel(x: np.ndarray, m: float, y0: float) -> np.ndarray:
    """
    Implements a linear model.

    Parameters:
        x (np.ndarray): The x values.
        m (float): The slope of the line.
        y0 (float): The y-intercept of the line.
    Returns:
        np.ndarray: The y values.
    """
    return m*x+y0

def single_file_analysis(filepath: str, number_of_bins: int, savefig: bool=False, error: str='std') -> tuple:
    """
    Performs a signal over radius analysis on a single file/experiment run.

    Parameters:
        filepath (str): The path to the file.
        number_of_bins (int): The number of radius bins to use.
        savefig (bool): Whether to save the plot.
        error (str): The type of error to use.
    Returns:
        tuple: A tuple containing the binned photometry data, the standard error of the mean, the standard deviation, and the mouse
        id.
    """
    mouse_id = extract_mouse_id(os.path.basename(filepath), mouse_id_length)
    current_data_dataframe = pd.read_csv(filepath, sep=',', header=None)
    current_data_array = current_data_dataframe.to_numpy()
    radius_data_array = current_data_array[:,radius_row]
    photometry_array_1 = current_data_array[:,signal_1_row]
    bin_tresholds = np.linspace(0., 1., number_of_cuts)
    binned_radii = np.full((number_of_bins), np.nan)
    binned_photometry_1 = np.full((number_of_bins), np.nan)
    binned_photometry_1_std = np.full((number_of_bins), np.nan)
    binned_photometry_1_sem = np.full((number_of_bins), np.nan)
    for i in range(number_of_bins):
        if(radius_data_array[np.logical_and(radius_data_array > bin_tresholds[i], radius_data_array < bin_tresholds[i+1])].size > 0):
            binned_radii[i] = np.mean(radius_data_array[np.logical_and(radius_data_array > bin_tresholds[i], radius_data_array < bin_tresholds[i+1])])
            binned_photometry_1[i] = np.mean(photometry_array_1[np.logical_and(radius_data_array > bin_tresholds[i], radius_data_array < bin_tresholds[i+1])])
            sqrt_of_sample_size = np.sqrt(photometry_array_1[np.logical_and(radius_data_array > bin_tresholds[i], radius_data_array < bin_tresholds[i+1])].size)
            binned_photometry_1_std[i] = np.std(photometry_array_1[np.logical_and(radius_data_array > bin_tresholds[i], radius_data_array < bin_tresholds[i+1])])
            binned_photometry_1_sem[i] = binned_photometry_1_std[i]/sqrt_of_sample_size
    
    return binned_photometry_1, binned_photometry_1_sem, binned_photometry_1_std, mouse_id

def badgeAnalysis(folderpath: str, number_of_bins: int, savefig: bool=False, error: str='std') -> tuple:
    """
    Performs a signal over radius analysis on a folder containing multiple files/experiment runs.
    Also performs the single file analysis on each file.

    Parameters:
        folderpath (str): The path to the folder.
        number_of_bins (int): The number of radius bins to use.
        savefig (bool): Whether to save the plot.
        error (str): The type of error to use.
    Returns:
        tuple: A tuple containing the binned photometry data, the standard error of the mean, and the standard deviation.
    """
    list_of_files = [f for f in os.listdir(folderpath) if not f.startswith('.')]
    buffer_photometry_array = np.full((len(list_of_files), number_of_bins), np.nan)
    buffer_sem_array = np.full((len(list_of_files), number_of_bins), np.nan)
    buffer_std_array = np.full((len(list_of_files), number_of_bins), np.nan)
    mouse_id_array = np.full((len(list_of_files)), np.nan)
    for i, file in enumerate(list_of_files):
        file_path = os.path.join(folderpath, file)
        print(file_path)
        buffer_photometry_array[i], buffer_sem_array[i], buffer_std_array, mouse_id_array[i] = single_file_analysis(file_path, number_of_bins, error)
    final_photometry_array = np.nanmean(buffer_photometry_array, axis=0)
    final_std_array = np.nanstd(buffer_photometry_array, axis=0)
    final_sem_array = final_std_array/np.sqrt(len(list_of_files))
    return final_photometry_array, final_sem_array, final_std_array

#Script starts here:
for number_of_bins in list_of_wanted_bin_numbers:
    number_of_cuts = number_of_bins + 1
    bin_string = str(number_of_bins)
    testDataNSF_photometry, testDataNSF_sem, testDataNSF_std = badgeAnalysis(os.path.join(path_of_script, 'testData', 'NSF'), number_of_bins)
    testDataOF_photometry, testDataOF_sem, testDataOF_std = badgeAnalysis(os.path.join(path_of_script, 'testData', 'OF'), number_of_bins)
    if fit:
        popt_nsf, pcov_nsf = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), testDataNSF_photometry, p0=[testDataNSF_photometry[-1]-testDataNSF_photometry[0], testDataNSF_photometry[0]], sigma=testDataNSF_std)
        popt_of, pcov_of = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), testDataOF_photometry, p0=[testDataOF_photometry[-1]-testDataOF_photometry[0], testDataOF_photometry[0]], sigma=testDataOF_std)
    
    np.savetxt(os.path.join(output_dir, bin_string+'_NSF.txt'), np.transpose(np.array([testDataNSF_photometry, testDataNSF_sem])), delimiter=',', header='mean, sem')
    np.savetxt(os.path.join(output_dir, bin_string+'_OF.txt'), np.transpose(np.array([testDataOF_photometry, testDataOF_sem])), delimiter=',', header='mean, sem')

    plt.plot(np.linspace(0., 1., number_of_bins), testDataNSF_photometry, label='NSF')
    plt.fill_between(np.linspace(0., 1., number_of_bins), testDataNSF_photometry-testDataNSF_sem, testDataNSF_photometry+testDataNSF_sem, alpha=0.2, label='NSF sem')
    plt.plot(np.linspace(0., 1., number_of_bins), testDataOF_photometry, label='OF')
    plt.fill_between(np.linspace(0., 1., number_of_bins), testDataOF_photometry-testDataOF_sem, testDataOF_photometry+testDataOF_sem, alpha=0.2, label='OF sem')

    if fit:
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_nsf), label='NSF Fit')
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_of), label='OF Fit')

        np.savetxt(os.path.join(output_dir, bin_string+'_nsf_fitparams.txt'), np.array([*popt_nsf, np.sqrt(pcov_nsf[0][0]), np.sqrt(pcov_nsf[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
        np.savetxt(os.path.join(output_dir, bin_string+'_of_fitparams.txt'), np.array([*popt_of, np.sqrt(pcov_of[0][0]), np.sqrt(pcov_of[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
    
    plt.title(bin_string)
    plt.ylim(-0.5, 1.)
    plt.legend()
    plt.savefig(os.path.join(output_dir, bin_string+'.svg'), transparent=True, format='svg')
    plt.clf()

