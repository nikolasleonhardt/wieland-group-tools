#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import warnings

#Time when the experiment shuts down regardless of bite
experiment_cutoff_time:int = 1200
#Length of the unique mouse identifier
mouse_id_length:int = 2
#Directory variables
outputAppendix = 'transformedData'
name_of_data_folder = 'testData/NSF'
#name_of_data_folder = 'testData/OF'

path_of_script = os.path.dirname(os.path.realpath(__file__))
outputpath = os.path.join(path_of_script, outputAppendix)
os.makedirs(outputpath, exist_ok=True)

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

def getValidIndices(minValue: float, maxValue: float, time_array: np.ndarray) -> tuple:
    """
    Finds the indices of the time array that are within the given time range.

    Parameters:
        minValue (float): The minimum time value.
        maxValue (float): The maximum time value.
        time_array (np.ndarray): The array of time values.
    
    Returns:
        tuple: A tuple containing the minimum and maximum indices.
    """
    minIndexToBeSet = True
    maxIndexToBeSet = True
    minIndex = 0
    maxIndex = 0
    for index, time in enumerate(time_array):
        if minIndexToBeSet and time >= minValue:
            minIndex = index
            minIndexToBeSet = False
        if maxIndexToBeSet and time >= maxValue:
            #Because this is meant for scliping the data, we want to include the last index and not use index-1.
            maxIndex = index
            maxIndexToBeSet = False
            break

    return minIndex, maxIndex

def determineFramerate(signal_times: np.ndarray, frames: np.ndarray) -> float:
    """
    Determines the frame rate of the camera based on the signal times and the number of frames.

    Parameters:
        signal_times (np.ndarray): The array of signal times.
        frames (np.ndarray): The array of camera frames.

    Returns:
        float: The frame rate of the camera.
    """
    last_signal_time = signal_times[-1]
    if last_signal_time == 0:
        raise ValueError('Signal times are invalid. The last signal time is 0.')
    
    number_of_frames = np.size(frames)
    if last_signal_time >= experiment_cutoff_time:
        framerate = number_of_frames/experiment_cutoff_time
    else:
        framerate = number_of_frames/last_signal_time
    return framerate

def binningToCameraResolution(signal_times: np.ndarray, signal1: np.ndarray, signal2: np.ndarray, camera_frames: np.ndarray, event_array: np.ndarray) -> tuple:
    """
    Bins multiple calcium sensor readings to the camera's lower frame rate.

    Parameters:
        signal_times (np.ndarray): The array of signal times.
        signal1 (np.ndarray): The array of the first calcium sensor readings.
        signal2 (np.ndarray): The array of the second calcium sensor readings.
        camera_frames (np.ndarray): The array of camera frames.
        event_array (np.ndarray): The array of event values.
    
    Returns:
        tuple: A tuple containing the camera frame times, the binned first signal data array, and the binned second signal data array.
    """
    final_signal1_data = np.zeros_like(camera_frames)
    final_signal2_data = np.zeros_like(camera_frames)
    final_signal1_data[:] = np.nan
    final_signal2_data[:] = np.nan
    first_camera_index = int(np.where(event_array == 1.0)[0][0])
    start_of_measurement_time = signal_times[first_camera_index]
    corrected_signal_times = signal_times - start_of_measurement_time
    camera_framerate = determineFramerate(corrected_signal_times, camera_frames)
    time_of_camera_frame = camera_frames/camera_framerate
    frame_time_cuts = time_of_camera_frame[:-1]+np.diff(time_of_camera_frame)/2
    for index, frame in enumerate(camera_frames):
        if index == 0:
            masked_time_array = np.ma.masked_outside(corrected_signal_times, time_of_camera_frame[index], frame_time_cuts[index])
        elif index == np.size(time_of_camera_frame)-1:
            masked_time_array = np.ma.masked_outside(corrected_signal_times, frame_time_cuts[-1], time_of_camera_frame[-1])
        else:
            masked_time_array = np.ma.masked_outside(corrected_signal_times, frame_time_cuts[index-1], frame_time_cuts[index])
            
        mask = masked_time_array.mask
        masked_signal1 = np.ma.masked_array(signal1, mask)
        masked_signal2 = np.ma.masked_array(signal2, mask)
        final_signal1_data[index] = np.mean(signal1[~mask])
        final_signal2_data[index] = np.mean(signal2[~mask])

    print('New set of times:')
    print(corrected_signal_times[-1])
    print(time_of_camera_frame[-1])

    if np.isnan(np.sum(final_signal1_data)) or np.isnan(np.sum(final_signal2_data)):
        warnings.warn('Something has gone wrong. There is a NaN in the transformed signal values!')
    return time_of_camera_frame, final_signal1_data, final_signal2_data

def single_file_analysis(location_path: str, signal_path: str) -> np.ndarray:
    """
    Reads out the data from given location and signal data files, bins the signal data to the camera frame rate and transforms the x/y coordinates to polar coordinates.
    The resulting data is then returned as a numpy array in the following format:
    time, radius, polar angle, signal 1 data, signal 2 data
     .  ,   .   ,      .     ,       .       ,       .
     .  ,   .   ,      .     ,       .       ,       .  
     .  ,   .   ,      .     ,       .       ,       .   
    
    Parameters:
        location_path (str): The path to the location data file.
        signal_path (str): The path to the signal data file.

    Returns:
        np.ndarray: The transformed data in the format described above.
    """
    location_data = pd.read_csv(location_path)
    signal_data = pd.read_csv(signal_path)
    location_array = location_data.to_numpy()
    signal_array = signal_data.to_numpy()
    location_array = location_array.transpose()
    signal_array = signal_array.transpose()
    time_values_sensor = np.array(signal_array[0], np.float64)
    sensor1 = np.array(signal_array[1], np.float64)
    sensor2 = np.array(signal_array[2], np.float64)
    event_array = np.array(signal_array[3], int)
    frame_number = np.array(location_array[6], np.float64)
    x_values = np.array(location_array[7], np.float64)
    y_values = np.array(location_array[8], np.float64)
    #Shift the x/y coordinate values so that the center of the arena is the (0,0) point
    min_x = np.min(x_values)
    min_y = np.min(y_values)
    x_values = x_values-min_x
    y_values = y_values-min_y
    max_x = np.max(x_values)
    max_y = np.max(y_values)
    center_x = max_x/2
    center_y = max_y/2
    x_values = x_values-center_x
    y_values = y_values-center_y
    #Transform the x/y coordinates to 2D polar coordinates
    r_values = np.sqrt(np.square(x_values) + np.square(y_values))
    phi_values = np.arctan2(y_values, x_values)
    #and normalize the radius
    normed_r_values = r_values/np.max(r_values)
    #We need the function below, because the camera frame rate is lower than the sensor frame rate 
    final_time, final_data_1, final_data_2 = binningToCameraResolution(time_values_sensor, sensor1, sensor2, frame_number, event_array)
    return np.array(np.stack([final_time, normed_r_values, phi_values, final_data_1, final_data_2], axis=1))

#Script starts here:
#Get the paths to the original signal and location files respectively 
signal_file_paths = glob.glob('*.csv', root_dir=os.path.join(path_of_script, name_of_data_folder, 'signal'))
location_file_paths = glob.glob('*.csv', root_dir=os.path.join(path_of_script, name_of_data_folder, 'location'))

#Create empty array to store the mouse ids of the various files
mouse_id_array_signals = np.zeros(np.size(signal_file_paths))
mouse_id_array_locations = np.zeros(np.size(location_file_paths))
#Fill the arrays with the mouse ids corresponding to the files
for index, path in enumerate(signal_file_paths):
    mouse_id_array_signals[index] = int(extract_mouse_id(signal_file_paths[index], mouse_id_length))
for index, path in enumerate(location_file_paths):
    mouse_id_array_locations[index] = int(extract_mouse_id(location_file_paths[index], mouse_id_length))

#Loop over all location files
for index, mouse_id in enumerate(mouse_id_array_locations):
    #Check if there is a signal file corresponding to the location file
    if mouse_id in mouse_id_array_signals:
        #Get the index of the corresponding signal file
        signal_index = int(np.where(mouse_id_array_signals==mouse_id)[0])
        #Get the paths to both files
        current_signal_path = os.path.join(path_of_script, name_of_data_folder, 'signal', signal_file_paths[signal_index])
        current_location_path = os.path.join(path_of_script, name_of_data_folder, 'location', location_file_paths[index])
        #Transform the data for this experiment run to polar coordinates and match frame rates
        final_data_array = single_file_analysis(current_location_path, current_signal_path)
        #Save the transformed data to a .txt/.csv file
        transposed_final_data = final_data_array.transpose()
        np.savetxt(os.path.join(outputpath, str(int(mouse_id))+'.txt'), final_data_array, fmt='%1.14f', delimiter=',')
    else:
        warnings.warn(f"There is a signal file missing for mouse id: {mouse_id}")   