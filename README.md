# wieland-group-tools
Collection of various tools developed for and used in the Wieland group at Heidelberg University.

## 1. System requirements
### 1.1 OS

These scripts should function on any operating system that can run Python 3 and the required libraries listed below.
The scripts have been tested on a Macbook Air M2 (16GB) running macOS Sequoia 15.3.

### 1.2 Python

This software was tested using Python 3.13.1. However it should also work for other versions of python that support the required python libraries listed below.

### 1.3 Python libraries

The library requirements for these scripts can also be found in the 'requirements.txt' file, but they are also listed below:

1. Numpy version 1.24.3
2. Matplotlib version 3.7.1
3. Pandas version 2.2.3
4. Scipy version 1.15.1

It is very likely that configurations that are not too far from the versions listed above will work just fine, but no guarantee can be made.
### 1.4 Non-standard hardware
No non-standard hardware is required to run the scripts.
## 2. Installation guide
### 2.1 Python 3
First, install Python 3 to run the scripts. The version we used, can be found [here](https://www.python.org/downloads/release/python-3131/).
Follow the instructions given there to install Python. 
### 2.2 Repository
Clone the repository into a directory of your choice, or simply download the files and save them anywhere on your computer.
### 2.3 Python libraries
Our script uses a variety of publically available python libraries, listed under **1.3 Python libraries**. You can either install them manually or simply use the supplemented *requirements.txt* file with pip via the command:
> pip install -r path/to/requirements.txt
### 2.4 Install time
All in all the installation should not take longer than 10 minutes on a "*normal*" desktop computer. However, this can of course vary, depending on internet speed and CPU power.

## 3. Demo and instructions for use

### 3.1 Test data and demo

For demo purposes a small set of test data is provided in the GitHub repository under "testData/originalData". The signal and location data is originally saved in separate files. 
To combine the data into a single file, bin the signal data to the camera frame rate and transform to polar coordinates one can run the "transform_data.py" script. The script works out of the box with the test data provided.
The script then creates a directory named "transformedData", where the new transformed data files are stored. The data structure in these files is as following:
    time, radius, polar angle, signal 1 data, signal 2 data 

We use this file format in our further analysis.

After the original data has been "transformed", one can run the "heatmaps.py" script to obtain heatmaps representing the probability density of the mouse in the arena. Again, this script should work out of the box on the provided test data once it has been preprocessed with the previous script. The resulting heatmaps can then be found in a directory called "heatmaps/".

### 3.2 Expected run time

On a Macbook Air M2 (16GB) the "transform_data.py" script with the test data took 75.0 s to run and the "heatmaps.py" script 12.6 s.
Of course run times can vary from machine to machine.

## 3.3 Further instructions for use

The scripts were created and used in a UNIX environment. For use on a Windows machine it might be necessary to alter the nomenclature of the paths in the constants section of both scripts. Additionally of course, please feel free to also alter other constants or experiment with the other functions provided. If questions arise, please feel free to contact us through GitHub.
