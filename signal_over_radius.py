import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from scipy.optimize import curve_fit

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

def getRadiusAverage(input_array:np.array):
    output_array = np.zeros(np.size(input_array)-1)
    for i in range(np.size(input_array)-1):
        output_array[i] = (input_array[i]+input_array[i+1])/2
    return output_array

def linearModel(x, m, y0):
    return m*x+y0

def constructSebastiansMasterCSVTable(mean_array, sem_array, mouse_id_array):
    double_mouse_id_array = np.repeat(mouse_id_array, 2)
    master_array = np.full((np.size(mean_array, axis=1)+1, np.size(double_mouse_id_array)), np.nan)
    master_array[0][:] = double_mouse_id_array
    for j in range(np.size(double_mouse_id_array)):
        if j % 2 == 0:
            master_array[1:, j] = mean_array[int(j/2)]
        else:
            master_array[1:, j] = sem_array[int((j-1)/2)]
    df = pd.DataFrame(master_array)
    return df

def single_file_analysis(filepath, number_of_bins, savefig=False, error='std'):
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

def badgeAnalysis(folderpath, number_of_bins, savefig=False, error='std'):
    list_of_files = [f for f in os.listdir(folderpath) if not f.startswith('.')]
    buffer_photometry_array = np.full((len(list_of_files), number_of_bins), np.nan)
    buffer_sem_array = np.full((len(list_of_files), number_of_bins), np.nan)
    buffer_std_array = np.full((len(list_of_files), number_of_bins), np.nan)
    mouse_id_array = np.full((len(list_of_files)), np.nan)
    for i, file in enumerate(list_of_files):
        file_path = os.path.join(folderpath, file)
        print(file_path)
        buffer_photometry_array[i], buffer_sem_array[i], buffer_std_array, mouse_id_array[i] = single_file_analysis(file_path, number_of_bins, error)
    #master_df = constructSebastiansMasterCSVTable(buffer_photometry_array, buffer_error_array, mouse_id_array)
    #master_df.to_csv(os.path.join(folderpath, 'master.csv'))
    final_photometry_array = np.nanmean(buffer_photometry_array, axis=0)
    final_std_array = np.nanstd(buffer_photometry_array, axis=0)
    final_sem_array = final_std_array/np.sqrt(len(list_of_files))
    return final_photometry_array, final_sem_array, final_std_array

path_of_script = os.path.dirname(os.path.realpath(__file__))
number_of_measurement_rows:int = 5
frame_time_row:int = 0
radius_row:int = 1
angle_row:int = 2
signal_1_row:int = 3
signal_2_row:int = 4
output_dir:str = '/Users/nikolasleonhardt/Documents/NeuroAna/NSFonlyMiceEverywhereInArena/signalOverRadius'
mouse_id_length:int = 2

os.makedirs(output_dir, exist_ok=True)

list_of_wanted_bin_numbers = [10]
fit = True
for number_of_bins in list_of_wanted_bin_numbers:
    number_of_cuts = number_of_bins + 1
    bin_string = str(number_of_bins)
    ghsham_photometry, ghsham_sem, ghsham_std = badgeAnalysis(os.path.join(path_of_script, 'fGHSham'), number_of_bins)
    ghsni_photometry, ghsni_sem, ghsni_std = badgeAnalysis(os.path.join(path_of_script, 'fGHSNI'), number_of_bins)
    sisham_photometry, sisham_sem, sisham_std = badgeAnalysis(os.path.join(path_of_script, 'fSISham'), number_of_bins)
    sisni_photometry, sisni_sem, sisni_std = badgeAnalysis(os.path.join(path_of_script, 'fSISNI'), number_of_bins)

    if fit:
        popt_ghsham, pcov_ghsham = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), ghsham_photometry, p0=[ghsham_photometry[-1]-ghsham_photometry[0], ghsham_photometry[0]], sigma=ghsham_std)
        popt_ghsni, pcov_ghsni = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), ghsni_photometry, p0=[ghsni_photometry[-1]-ghsni_photometry[0], ghsni_photometry[0]], sigma=ghsni_std)
        popt_sisham, pcov_sisham = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), sisham_photometry, p0=[sisham_photometry[-1]-sisham_photometry[0], sisham_photometry[0]], sigma=sisham_std)
        popt_sisni, pcov_sisni = curve_fit(linearModel, np.linspace(0., 1., number_of_bins), sisni_photometry, p0=[sisni_photometry[-1]-sisni_photometry[0], sisni_photometry[0]], sigma=sisni_std)
    
    np.savetxt(os.path.join(output_dir, bin_string+'_ghsham.txt'), np.transpose(np.array([ghsham_photometry, ghsham_sem])), delimiter=',', header='mean, sem')
    np.savetxt(os.path.join(output_dir, bin_string+'_ghsni.txt'), np.transpose(np.array([ghsni_photometry, ghsni_sem])), delimiter=',', header='mean, sem')
    np.savetxt(os.path.join(output_dir, bin_string+'_sisham.txt'), np.transpose(np.array([sisham_photometry, sisham_sem])), delimiter=',', header='mean, sem')
    np.savetxt(os.path.join(output_dir, bin_string+'_sisni.txt'), np.transpose(np.array([sisni_photometry, sisni_sem])), delimiter=',', header='mean, sem')

    plt.plot(np.linspace(0., 1., number_of_bins), ghsham_photometry, label='fGHSham')
    plt.fill_between(np.linspace(0., 1., number_of_bins), ghsham_photometry-ghsham_sem, ghsham_photometry+ghsham_sem, alpha=0.2, label='fGHSham sem')
    plt.plot(np.linspace(0., 1., number_of_bins), ghsni_photometry, label='fGHSNI')
    plt.fill_between(np.linspace(0., 1., number_of_bins), ghsni_photometry-ghsni_sem, ghsni_photometry+ghsni_sem, alpha=0.2, label='fGHSNI sem')
    plt.plot(np.linspace(0., 1., number_of_bins), sisham_photometry, label='fSISham')
    plt.fill_between(np.linspace(0., 1., number_of_bins), sisham_photometry-sisham_sem, sisham_photometry+sisham_sem, alpha=0.2, label='fSISham sem')
    plt.plot(np.linspace(0., 1., number_of_bins), sisni_photometry, label='fSISNI')
    plt.fill_between(np.linspace(0., 1., number_of_bins), sisni_photometry-sisni_sem, sisni_photometry+sisni_sem, alpha=0.2, label='fSISNI sem')

    if fit:
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_ghsham), label='fGHSham Fit')
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_ghsni), label='fGHSNI Fit')
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_sisham), label='fSISham Fit')
        plt.plot(np.linspace(0., 1., number_of_bins), linearModel(np.linspace(0., 1., number_of_bins), *popt_sisni), label='fSISNI Fit')

        np.savetxt(os.path.join(output_dir, bin_string+'_ghsham_fitparams.txt'), np.array([*popt_ghsham, np.sqrt(pcov_ghsham[0][0]), np.sqrt(pcov_ghsham[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
        np.savetxt(os.path.join(output_dir, bin_string+'_ghsni_fitparams.txt'), np.array([*popt_ghsni, np.sqrt(pcov_ghsni[0][0]), np.sqrt(pcov_ghsni[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
        np.savetxt(os.path.join(output_dir, bin_string+'_sisham_fitparams.txt'), np.array([*popt_sisham, np.sqrt(pcov_sisham[0][0]), np.sqrt(pcov_sisham[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
        np.savetxt(os.path.join(output_dir, bin_string+'_sisni_fitparams.txt'), np.array([*popt_sisni, np.sqrt(pcov_sisni[0][0]), np.sqrt(pcov_sisni[1][1])]), delimiter=',', header='m, y0, delta m, delta y0')
    
    plt.title(bin_string)
    plt.ylim(-0.5, 1.)
    plt.legend()
    plt.savefig(os.path.join(output_dir, bin_string+'.svg'), transparent=True, format='svg')
    plt.clf()

