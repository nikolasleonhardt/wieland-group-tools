import os 
import numpy as np
import pandas as pd

def extractMouseId(filename, mouse_id_length):
    mouse_id = ''
    count = 0
    for char in filename:
        if char.isdigit():
            mouse_id += char
            count += 1
            if count == mouse_id_length:
                break
    return mouse_id

def generateMouseIDList(fileList):
    mouseIDList = []
    for file in fileList:
        mouseID = extractMouseId(file, 2)
        if mouseID not in mouseIDList:
            mouseIDList.append(mouseID)
    return mouseIDList

def extractFilamentNumber(filename):
    dash_index = filename.find('-')
    filamentString = filename[dash_index-1:-4]
    filamentString = filamentString.replace('-', '.')
    filamentString = filamentString.replace('g', '')
    return float(filamentString)

def sortMouseFiles(mouseFiles):
    sortedMouseFiles = []
    lowestFilament = np.inf
    while len(mouseFiles) > 0:
        for mouseFile in mouseFiles:
            currentFilament = extractFilamentNumber(mouseFile)
            if currentFilament < lowestFilament:
                currentLowestFile = mouseFile
                lowestFilament = currentFilament
        sortedMouseFiles.append(currentLowestFile)
        lowestFilament = np.inf
        mouseFiles.remove(currentLowestFile)
    return sortedMouseFiles

def concatenateFiles(fileList):
    mouseIDList = generateMouseIDList(fileList)
    for mouseID in mouseIDList:
        mouseFiles = [file for file in fileList if mouseID == extractMouseId(file, 2)]
        numberOfFilaments = len(mouseFiles)
        filamentSortedMouseFiles = sortMouseFiles(mouseFiles)
        dataframeList = []
        for file in filamentSortedMouseFiles:
            dataframeList.append(pd.read_csv(os.path.join(inputDir, file)))    
        mouseDataframe = pd.concat(dataframeList, ignore_index=True, sort=False)
        mouseDataframe.to_csv(os.path.join(inputDir, mouseID+'.csv'), index=False, float_format='%.4f')

def addFilamentInformationToCSV(fileList):
    for file in fileList:
        filepath = os.path.join(inputDir, file)
        fileFilament = extractFilamentNumber(file)
        df = pd.read_csv(filepath)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.assign(filament = float(fileFilament))
        df.to_csv(path_or_buf=filepath, index=False, float_format='%.4f')
        print(filepath)

inputDir = r'/Users/nikolasleonhardt/Documents/NeuroAna/vonFreyDritterAnlauf/fSISNI'

fileList = [file for file in os.listdir(inputDir) if file.endswith('.csv')]

#addFilamentInformationToCSV(fileList)
concatenateFiles(fileList)