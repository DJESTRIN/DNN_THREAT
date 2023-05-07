# Command should be python classify_observation.py
import subprocess
import os
import sys
import h5py
import numpy as np


class Classify_observation:
    # INITIALIZER
    # pathToIlastik is the path to the Ilastik location
    # projectFilepath is the filepath to the ilp file
    # exportFolder is the folder you want to store
        #if exportFolder doesn't exist, it will be created
    def __init__(self,pathToIlastik, projectFilePath,exportFolder, exportTextFile):
        self.pathToIlastik = pathToIlastik
        self.projectFilePath = projectFilePath
        self.current_dir = os.getcwd()
        self.exportFolder = exportFolder
        self.exportTextFile = exportTextFile
        # If the export folder doesn't exist, create it
        if not os.path.exists(self.exportFolder):
            os.mkdir(self.exportFolder)

    # produces the export h5 files and places them in the exportFolder based on input files in inputDirPath
    # inputDirPath is a path to the directory that contains the input files
    def exportClassesFromDir(self, inputDirPath):
        inputFiles = [inputDirPath + "\\" + f for f in os.listdir(inputDirPath) if f.endswith('.png')]
        self.exportClassesFromFiles(inputFiles)

    # produces the export h5 files and places them in the exportFolder based on an arr of inputFilePaths
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

    def displayAllClassesInOutputTextFile(self):
        exportFolderFiles = os.listdir(self.exportFolder)
        # outputFiles = [f for f in os.listdir(self.exportFolder) if f.endswith('.h5')]
        for file in exportFolderFiles:
            if file.endswith('.h5'):
                with h5py.File("{}\{}".format(self.exportFolder, file), 'r') as f:
                    for dataset_name in f: #The dataset is 'exported_data', but just in case that doesn't work we use this
                        dset = f[dataset_name]
                        nparr = np.array(dset).flatten()
                        with open(self.exportTextFile, "a") as textfile:
                            textfile.write("Values for {}: {}\n".format(file, np.unique(nparr)))
                            os.remove("{}\{}".format(self.exportFolder,file))
        with open(self.exportTextFile, "a") as textfile:
            textfile.write("\n\n---------------\n\n")