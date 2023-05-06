# Command should be python classify_observation.py
import subprocess
import os
import sys
import h5py
import numpy as np


class classify_observation:
    def __init__(self,pathToIlastik, projectFilePath,exportFolder):
        self.pathToIlastik = pathToIlastik
        self.projectFilePath = projectFilePath
        self.current_dir = os.getcwd()
        self.exportFolder = exportFolder
        # If the export folder doesn't exist, create it
        if not os.path.exists(self.exportFolder):
            os.mkdir(self.exportFolder)

    # inputDir is a directory that contains the input files
    def exportClassesFromDir(self, inputDir):
        inputFiles = [inputDir + "\\" + f for f in os.listdir(inputDir) if f.endswith('.png')]
        self.exportClassesFromFiles(inputFiles)

    # inputFilePaths is an array of file paths
    def exportClassesFromFiles(self, inputFilePaths):
        # operation takes place at the Ilastik file location
        os.chdir(self.pathToIlastik)
        inputFilesPathsArr = []
        for i in range(len(inputFilePaths)):
            inputFilesPathsArr.append(self.current_dir + '\\' + inputFilePaths[i])
        inputFilesPathsArr = ' '.join(inputFilesPathsArr)
        command = "ilastik.exe --headless --project={}\{} --export_source=\"Simple Segmentation\" --output_filename_format={}/{}/{}_{}.h5 {}".format(self.current_dir, self.projectFilePath, self.current_dir, self.exportFolder, '{nickname}', '{result_type}', inputFilesPathsArr)
        

        # Windows has a limit of about 8000 characters per command, so we divide the command in half if it is over
        if (len(command) > 8000):
            half_len = len(inputFilePaths) // 2
            self.exportClassesFromFiles(inputFilePaths[:half_len])
            self.exportClassesFromFiles(inputFilePaths[half_len:])
        else:
            # brings the os back to the original directory
            os.system(command)
            os.chdir(self.current_dir)
            self.getAllClassesInOutput()

    def getAllClassesInOutput(self):
        outputFiles = [f for f in os.listdir(self.exportFolder) if f.endswith('.h5')]
        for file in outputFiles:
            with h5py.File("{}\{}".format(self.exportFolder, file), 'r') as f:
                for dataset_name in f: #The dataset is 'exported_data', but just in case that doesn't work we use this
                    dset = f[dataset_name]
                    nparr = np.array(dset).flatten()
                    with open("classifying.txt", "a") as textfile:
                        textfile.write("Values for {}: {}\n".format(file, np.unique(nparr)))
            os.remove("Exports\{}".format(file))


# classify_observation's init provides the outputs
# Format of parameters:
# (Path to Ilastik, Path to project file, path to export folder)
x = classify_observation("C:\Program Files\ilastik-1.4.0", "BattleZoneVersion0.ilp","Exports/")
    
# getAllClassesInOutput records the different classes in integer form on the textfile "classifying.txt"

# x.exportClassesFromFiles(["BattleZoneImageFolder\BattleZone-v4Image1438.png", "BattleZoneImageFolder\BattleZone-v4Image1330.png", "BattleZoneImageFolder\BattleZone-v4Image1322.png"])
x.exportClassesFromDir("BattleZoneImageFolder")