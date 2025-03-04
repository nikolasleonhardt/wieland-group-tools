import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

def cleanUpDioArray(boolArray):
    cleanedArray = np.zeros_like(boolArray)
    for i in range(np.size(boolArray)-2):
        if boolArray[i] == 0 and boolArray[i+1] == 1:
            cleanedArray[i+1] = 1
    return cleanedArray

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

def generatePokeAUCandAmplitude(fileList):
    #frame 0 is the begin of the baseline, which lasts for baselineDur
    #frame 
    baselineBeginFrame = 0
    baselineEndFrame = int(framerate*baselineDurationSec)
    pokeFrame = int(framerate*(timeBeforePokeSec+baselineDurationSec))
    endFrame = int(framerate*(baselineDurationSec+timeBeforePokeSec+timeAfterPokeSec))

    numberOfFramesAfterPoke = int(framerate*timeAfterPokeSec)

    finalMeanData = np.full((len(fileList), endFrame), np.nan)
    finalSemData = np.full((len(fileList), endFrame), np.nan)

    finalIndividualMouseArray = np.full((len(fileList), 5), np.nan)
    for mouseIndex, file in enumerate(fileList):
        print(file)
        mouseID = extractMouseId(file, 2)
        df = pd.read_csv(os.path.join(inputDir, file))
        signalArray = df[signalColumn].to_numpy()
        timeArray = df[timeColumn].to_numpy()
        dioArray = df[dioColumn].to_numpy()
        pokeWithWithdrawalBoolArray = df[pokeWithWithdrawalColumn].to_numpy()
        pokeWithNoWithdrawalBoolArray = df[pokeWithNoWithdrawalColumn].to_numpy()
        filamentArray = df[filamentColumn].to_numpy()
        maxFilamentIndex = np.argmax(filamentArray > filamentCutoff)
        signalArray = signalArray[:maxFilamentIndex]
        timeArray = timeArray[:maxFilamentIndex]
        dioArray = dioArray[:maxFilamentIndex]
        dioArray = cleanUpDioArray(dioArray)
        numberOfPokes = np.sum(dioArray)
        pokeWithWithdrawalBoolArray = pokeWithWithdrawalBoolArray[:maxFilamentIndex]
        pokeWithNoWithdrawalBoolArray = pokeWithNoWithdrawalBoolArray[:maxFilamentIndex]
        pokeIndices = dioArray.nonzero()[0]
        withdrawalPokes = pokeWithWithdrawalBoolArray[pokeIndices].nonzero()[0]
        nonWithdrawalPokes = pokeWithNoWithdrawalBoolArray[pokeIndices].nonzero()[0]
        if withdrawal:
            relevantPokeArray = withdrawalPokes
            mousePokeData = np.full((np.size(withdrawalPokes), endFrame), np.nan)
        else:
            relevantPokeArray = nonWithdrawalPokes
            mousePokeData = np.full((np.size(nonWithdrawalPokes), endFrame), np.nan)
        for pokeIndex, pokeNumber in enumerate(relevantPokeArray):
            pokeDataIndex = pokeIndices[pokeNumber]
            signalArrayPokeExerpt = signalArray[pokeDataIndex-pokeFrame: pokeDataIndex+numberOfFramesAfterPoke]
            baseline = np.mean(signalArrayPokeExerpt[baselineBeginFrame:baselineEndFrame])
            mousePokeData[pokeIndex] = signalArrayPokeExerpt-baseline

        meanMousePokeData = np.mean(mousePokeData, axis=0)
        semMousePokeData = np.std(mousePokeData, axis=0)/np.sqrt(np.size(mousePokeData, axis=0))
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2

        aucEndIndex = endFrame
        meanMouseDiff = np.diff(meanMousePokeData, prepend=0)
        if meanMousePokeData[pokeFrame+pokeBuffer] < 0:
            boolArrayOfValuesLargerZero = meanMousePokeData[pokeFrame+pokeBuffer:] > 0
            if any(boolArrayOfValuesLargerZero):
                aucEndIndex = np.argmax(boolArrayOfValuesLargerZero) + pokeFrame + pokeBuffer
        elif meanMousePokeData[pokeFrame+pokeBuffer] >= 0:
            boolArrayOfValuesSmallerZero = meanMousePokeData[pokeFrame+pokeBuffer:] < 0
            if any(boolArrayOfValuesSmallerZero):
                aucEndIndex = np.argmax(boolArrayOfValuesSmallerZero) + pokeFrame + pokeBuffer

        mouseAuc = np.sum(meanMousePokeData[pokeFrame:aucEndIndex])
        mouseAucLowEstimate = np.sum(meanMousePokeData[pokeFrame:aucEndIndex]-semMousePokeData[pokeFrame:aucEndIndex])
        mouseAucHighEstimate = np.sum(meanMousePokeData[pokeFrame:aucEndIndex]+semMousePokeData[pokeFrame:aucEndIndex])
        mouseAucError = (np.abs(mouseAuc-mouseAucLowEstimate)+np.abs(mouseAuc-mouseAucHighEstimate))/2

        mouseAmplitude = np.max(meanMousePokeData[pokeFrame:aucEndIndex])
        mouseAmplitudeLowEstimate = np.max(meanMousePokeData[pokeFrame:aucEndIndex]-semMousePokeData[pokeFrame:aucEndIndex])
        mouseAmplitudeHighEstimate = np.max(meanMousePokeData[pokeFrame:aucEndIndex]+ semMousePokeData[pokeFrame:aucEndIndex])
        mouseAmplitudeError = (np.abs(mouseAmplitude-mouseAmplitudeLowEstimate)+np.abs(mouseAmplitude-mouseAmplitudeHighEstimate))/2

        labels = []
        labels.append(r'$AUC = {value} \pm {error}$'.format(value=np.around(mouseAuc, 2), error=np.around(mouseAucError, 2)))
        labels.append(r'$amplitude = {value} \pm {error}$'.format(value=np.around(mouseAmplitude, 2), error=np.around(mouseAmplitudeError, 2)))


        plt.plot(np.arange(endFrame), meanMousePokeData, label='Mean')
        plt.fill_between(np.arange(endFrame), meanMousePokeData-semMousePokeData, meanMousePokeData+semMousePokeData, alpha=0.2, label='SEM')
        plt.fill_between(np.arange(start=pokeFrame, stop=aucEndIndex), np.zeros_like(np.arange(start=pokeFrame, stop=aucEndIndex)), meanMousePokeData[pokeFrame:aucEndIndex], color='pink', alpha=0.2)
        plt.vlines([pokeFrame, baselineEndFrame, aucEndIndex], ymin=-1, ymax=1, colors=['red', 'green', 'purple'], label='Poke')
        plt.hlines(0, 1, endFrame, colors='blue', linestyles='dashed')
        plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.7, handlelength=0, handletextpad=0)
        if withdrawal:
            plt.savefig(os.path.join(inputDir, file.replace('.csv', 'withdrawal.svg')), transparent=True, format='svg')
        else:
            plt.savefig(os.path.join(inputDir, file.replace('.csv', 'no_withdrawal.svg')), transparent=True, format='svg')
        plt.clf()
        finalMeanData[mouseIndex] = meanMousePokeData
        finalSemData[mouseIndex] = semMousePokeData

        finalIndividualMouseArray[mouseIndex] = np.array([mouseID, mouseAuc, mouseAucError, mouseAmplitude, mouseAmplitudeError])
    
    finalMiceMeanData = np.nanmean(finalMeanData, axis=0)
    finalMiceSemData = np.nanstd(finalMeanData, axis=0)/np.sqrt(len(fileList))

    aucEndIndex = endFrame
    meanMouseDiff = np.diff(finalMiceMeanData, prepend=0)
    if finalMiceMeanData[pokeFrame+pokeBuffer] < 0:
        boolArrayOfValuesLargerZero = finalMiceMeanData[pokeFrame+pokeBuffer:] > 0
        if any(boolArrayOfValuesLargerZero):
            aucEndIndex = np.argmax(boolArrayOfValuesLargerZero) + pokeFrame + pokeBuffer
    elif finalMiceMeanData[pokeFrame+pokeBuffer] >= 0:
        boolArrayOfValuesSmallerZero = finalMiceMeanData[pokeFrame+pokeBuffer:] < 0
        if any(boolArrayOfValuesSmallerZero):
            aucEndIndex = np.argmax(boolArrayOfValuesSmallerZero) + pokeFrame + pokeBuffer

    phenotypeAuc = np.sum(finalMiceMeanData[baselineEndFrame:aucEndIndex])
    phenotypeAucLowEstimate = np.sum(finalMiceMeanData[baselineEndFrame:aucEndIndex]-finalMiceSemData[baselineEndFrame:aucEndIndex])
    phenotypeAucHighEstimate = np.sum(finalMiceMeanData[baselineEndFrame:aucEndIndex]+finalMiceSemData[baselineEndFrame:aucEndIndex])
    phenotypeAucError = (np.abs(phenotypeAuc-phenotypeAucLowEstimate)+np.abs(phenotypeAuc-phenotypeAucHighEstimate))/2

    phenotypeAmplitude = np.max(finalMiceMeanData[pokeFrame:aucEndIndex])
    phenotypeAmplitudeLowEstimate = np.max(finalMiceMeanData[pokeFrame:aucEndIndex]-finalMiceSemData[pokeFrame:aucEndIndex])
    phenotypeAmplitudeHighEstimate = np.max(finalMiceMeanData[pokeFrame:aucEndIndex]+finalMiceSemData[pokeFrame:aucEndIndex])
    phenotypeAmplitudeError = (np.abs(phenotypeAmplitude-phenotypeAmplitudeHighEstimate)+np.abs(phenotypeAmplitude-phenotypeAmplitudeLowEstimate))/2   

    labels = []
    labels.append(r'$AUC = {value} \pm {error}$'.format(value=np.around(phenotypeAuc, 2), error=np.around(phenotypeAucError, 2)))
    labels.append(r'$amplitude = {value} \pm {error}$'.format(value=np.around(phenotypeAmplitude, 2), error=np.around(phenotypeAmplitudeError, 2)))

    plt.plot(np.arange(endFrame), finalMiceMeanData, label='Mean')
    plt.fill_between(np.arange(endFrame), finalMiceMeanData-finalMiceSemData, finalMiceMeanData+finalMiceSemData, alpha=0.2, label='SEM')
    plt.fill_between(np.arange(start=baselineEndFrame, stop=aucEndIndex), np.zeros_like(np.arange(start=baselineEndFrame, stop=aucEndIndex)), finalMiceMeanData[baselineEndFrame:aucEndIndex], color='pink', alpha=0.2)
    plt.vlines([pokeFrame, baselineEndFrame, aucEndIndex], ymin=-1, ymax=1, colors=['red', 'green', 'purple'], label=['Poke', 'Baseline End', 'aucEnd'])
    plt.hlines(0, 1, endFrame, colors='blue', linestyles='dashed')
    plt.legend(handles, labels, loc='best', fontsize='small', fancybox=True, framealpha=0.7, handlelength=0, handletextpad=0)
    if withdrawal:
        plt.savefig(os.path.join(inputDir, 'withdrawal.svg'), transparent=True, format='svg')
    else:
        plt.savefig(os.path.join(inputDir, 'no_withdrawal.svg'), transparent=True, format='svg')
    plt.clf()
    if withdrawal:
        np.savetxt(os.path.join(inputDir, 'withdrawalIndividualMice.txt'), finalIndividualMouseArray, delimiter=',', header='ID, AUC, AUCERR, AMP, AMPERR')
        np.savetxt(os.path.join(inputDir, 'withdrawalCombinedMice.txt'), np.array([phenotypeAuc, phenotypeAucError, phenotypeAmplitude, phenotypeAmplitudeError]), delimiter=',', header='AUC, AUCERR, AMP, AMPERR')
    else:
        np.savetxt(os.path.join(inputDir, 'noWithdrawalIndividualMice.txt'), finalIndividualMouseArray, delimiter=',', header='ID, AUC, AUCERR, AMP, AMPERR')
        np.savetxt(os.path.join(inputDir, 'noWithdrawalCombinedMice.txt'), np.array([phenotypeAuc, phenotypeAucError, phenotypeAmplitude, phenotypeAmplitudeError]), delimiter=',', header='AUC, AUCERR, AMP, AMPERR')
    return None

def withdrawalFrequency(folderList):
    listOfArrayWithWithdrawalCounts = []
    for folder in folderList:
        folderPath = os.path.join(inputDir, folder)
        fileList = [file for file in os.listdir(folderPath) if file.endswith('.csv')]
        withdrawalCountArray = np.full((len(fileList)), np.nan)
        finalFrequencyArray = np.full((2, len(fileList)), np.nan)
        for mouseIndex, file in enumerate(fileList):
            print(file)
            mouseID = extractMouseId(file, 2)
            df = pd.read_csv(os.path.join(folderPath, file))
            dioArray = df[dioColumn].to_numpy()
            pokeWithWithdrawalBoolArray = df[pokeWithWithdrawalColumn].to_numpy()
            pokeWithNoWithdrawalBoolArray = df[pokeWithNoWithdrawalColumn].to_numpy()
            filamentArray = df[filamentColumn].to_numpy()
            maxFilamentIndex = np.argmax(filamentArray > filamentCutoff)
            dioArray = dioArray[:maxFilamentIndex]
            dioArray = cleanUpDioArray(dioArray)
            numberOfPokes = np.sum(dioArray)
            pokeWithWithdrawalBoolArray = pokeWithWithdrawalBoolArray[:maxFilamentIndex]
            pokeWithNoWithdrawalBoolArray = pokeWithNoWithdrawalBoolArray[:maxFilamentIndex]
            pokeIndices = dioArray.nonzero()[0]
            withdrawalPokes = pokeWithWithdrawalBoolArray[pokeIndices].nonzero()[0]
            nonWithdrawalPokes = pokeWithNoWithdrawalBoolArray[pokeIndices].nonzero()[0]
            pokeFrequency = np.size(withdrawalPokes)/numberOfPokes
            withdrawalCountArray[mouseIndex] = pokeFrequency
            finalFrequencyArray[0, mouseIndex] = mouseID
            finalFrequencyArray[1, mouseIndex] = pokeFrequency
        listOfArrayWithWithdrawalCounts.append(withdrawalCountArray)
        plt.hist(withdrawalCountArray, label = folder, alpha=0.5)
        np.savetxt(os.path.join(outputDir, folder+'.txt'), np.transpose(finalFrequencyArray), delimiter=',', header='ID, frequency')
    plt.legend()
    plt.savefig(os.path.join(outputDir, 'frequencyHistogram.svg'), transparent=True, format='svg')
    plt.clf()
    return None

def cumAucOverWithdrawalFrequency(folderList):
    baselineBeginFrame = 0
    baselineEndFrame = int(framerate*baselineDurationSec)
    pokeFrame = int(framerate*(timeBeforePokeSec+baselineDurationSec))
    endFrame = int(framerate*(baselineDurationSec+timeBeforePokeSec+timeAfterPokeSec))
    numberOfFramesAfterPoke = int(framerate*timeAfterPokeSec)

    withdrawalFrequencyArrayList = []
    for folder in folderList:
        folderPath = os.path.join(inputDir, folder)
        fileList = [file for file in os.listdir(folderPath) if file.endswith('.csv')]

        finalIndividualCumMouseArray = np.full((2, len(fileList)+2), np.nan)
        withdrawalFrequencyArray = np.full((len(fileList)), np.nan)
        for mouseIndex, file in enumerate(fileList):
            mouseID = extractMouseId(file, 2)
            df = pd.read_csv(os.path.join(folderPath, file))
            signalArray = df[signalColumn].to_numpy()
            timeArray = df[timeColumn].to_numpy()
            dioArray = df[dioColumn].to_numpy()
            pokeWithWithdrawalBoolArray = df[pokeWithWithdrawalColumn].to_numpy()
            pokeWithNoWithdrawalBoolArray = df[pokeWithNoWithdrawalColumn].to_numpy()
            filamentArray = df[filamentColumn].to_numpy()
            maxFilamentIndex = np.argmax(filamentArray > filamentCutoff)
            signalArray = signalArray[:maxFilamentIndex]
            timeArray = timeArray[:maxFilamentIndex]
            dioArray = dioArray[:maxFilamentIndex]
            dioArray = cleanUpDioArray(dioArray)
            numberOfPokes = np.sum(dioArray)
            pokeWithWithdrawalBoolArray = pokeWithWithdrawalBoolArray[:maxFilamentIndex]
            pokeWithNoWithdrawalBoolArray = pokeWithNoWithdrawalBoolArray[:maxFilamentIndex]
            pokeIndices = dioArray.nonzero()[0]
            withdrawalPokes = pokeWithWithdrawalBoolArray[pokeIndices].nonzero()[0]
            nonWithdrawalPokes = pokeWithNoWithdrawalBoolArray[pokeIndices].nonzero()[0]
            pokeFrequency = np.size(withdrawalPokes)/numberOfPokes
            if withdrawalCheck:
                if withdrawal:
                    relevantPokeArray = withdrawalPokes
                    mousePokeData = np.full((np.size(withdrawalPokes), endFrame), np.nan)
                else:
                    relevantPokeArray = nonWithdrawalPokes
                    mousePokeData = np.full((np.size(nonWithdrawalPokes), endFrame), np.nan)
            else:
                relevantPokeArray = np.concatenate((withdrawalPokes, nonWithdrawalPokes))
                mousePokeData = np.full((np.size(relevantPokeArray), endFrame), np.nan)

            mouseCumAuc = 0
            for pokeIndex, pokeNumber in enumerate(relevantPokeArray):
                pokeDataIndex = pokeIndices[pokeNumber]
                signalArrayPokeExerpt = signalArray[pokeDataIndex-pokeFrame: pokeDataIndex+numberOfFramesAfterPoke]
                baseline = np.mean(signalArrayPokeExerpt[baselineBeginFrame:baselineEndFrame])
                signalArrayPokeExerpt = signalArrayPokeExerpt-baseline
                aucEndIndex = endFrame
                if signalArrayPokeExerpt[pokeFrame+pokeBuffer] < 0:
                    boolArrayOfValuesLargerZero = signalArrayPokeExerpt[pokeFrame+pokeBuffer:] > 0
                    if any(boolArrayOfValuesLargerZero):
                        aucEndIndex = np.argmax(boolArrayOfValuesLargerZero) + pokeFrame + pokeBuffer
                elif signalArrayPokeExerpt[pokeFrame+pokeBuffer] >= 0:
                    boolArrayOfValuesSmallerZero = signalArrayPokeExerpt[pokeFrame+pokeBuffer:] < 0
                    if any(boolArrayOfValuesSmallerZero):
                        aucEndIndex = np.argmax(boolArrayOfValuesSmallerZero) + pokeFrame + pokeBuffer
                
                pokeAuc = np.sum(signalArrayPokeExerpt[pokeFrame:aucEndIndex])
                mouseCumAuc += pokeAuc
                
            finalIndividualCumMouseArray[0, mouseIndex] = mouseID
            finalIndividualCumMouseArray[1, mouseIndex] = mouseCumAuc
            withdrawalFrequencyArray[mouseIndex] = pokeFrequency

        plt.scatter(withdrawalFrequencyArray, finalIndividualCumMouseArray[1, 0:len(fileList)], label=folder)
        finalIndividualCumMouseArray[0, -1] = np.mean(finalIndividualCumMouseArray[1, 0:len(fileList)])
        finalIndividualCumMouseArray[1, -1] = np.std(finalIndividualCumMouseArray[1, 0:len(fileList)])/np.sqrt(len(fileList))
        np.savetxt(os.path.join(outputDir, folder+'allPokes.txt'), np.transpose(finalIndividualCumMouseArray), delimiter=',', header='ID, cumulativeCA, last line corresponds phenotype mean, sem')    
    plt.ylim(-11000, 11000)
    plt.legend()
    plt.savefig(os.path.join(outputDir, 'withdrawals.svg'), transparent=True, format='svg')
    return None

inputDir = r'/Users/nikolasleonhardt/Documents/NeuroAna/vonFreyDritterAnlauf'
outputDir = r'/Users/nikolasleonhardt/Documents/NeuroAna/vonFreyDritterAnlauf/output'

signalColumn = 'sensor1_z_score_combined_std'
timeColumn = 'time_combined_std'
dioColumn = 'dio_checked'
pokeWithWithdrawalColumn = 'poke_with_withdrawel'
pokeWithNoWithdrawalColumn = 'poke_with_no_withdrawel'
filamentColumn = 'filament'
framerate = 120
baselineDurationSec = 1
timeBeforePokeSec = 0.5
timeAfterPokeSec = 6.5
filamentCutoff = 0.16
pokeBuffer = 10
withdrawal = False
withdrawalCheck = False

#fileList = [file for file in os.listdir(inputDir) if file.endswith('.csv')]
cumAucOverWithdrawalFrequency(['fGHSham', 'fGHSNI', 'fSISham', 'fSISNI'])