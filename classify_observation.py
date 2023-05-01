# Command should be python classify_observation.py
import subprocess
import os
import sys
import h5py
import numpy as np

# Get the current working directory
class classify_observation:
    def __init__(self,pathToIlastik, projectFilePath,exportFolder, inputFilePaths):
        self.current_dir = os.getcwd()
        os.chdir(pathToIlastik)
        inputFilesPathsArr = []
        for i in range(len(inputFilePaths)):
            inputFilesPathsArr.append(self.current_dir + '\\' + inputFilePaths[i])
        inputFilesPathsArr = ' '.join(inputFilesPathsArr)
        os.system("ilastik.exe --headless --project={}\{} --export_source=\"Simple Segmentation\" --output_filename_format={}/{}/{}_{}.h5 {}".format(self.current_dir, projectFilePath, self.current_dir, exportFolder, '{nickname}', '{result_type}', inputFilesPathsArr))
        os.chdir(self.current_dir)

    def getAllClassesInOutput(self):
        outputFiles = os.listdir("Exports")
        for file in outputFiles:
            with h5py.File("Exports\{}".format(file), 'r') as f:
                for dataset_name in f: #The dataset is 'exported_data', but just in case that doesn't work we use this
                    dset = f[dataset_name]
                    nparr = np.array(dset).flatten()
                    with open("classifying.txt", "a") as textfile:
                        textfile.write("Values for {}: {}\n".format(file, np.unique(nparr)))
            os.remove("Exports\{}".format(file))

# x = classify_observation("C:\Program Files\ilastik-1.4.0", "BattleZoneVersion0.ilp","Exports/" , ["BattleZoneImageFolder\BattleZone-v4Image14.png", "BattleZoneImageFolder\BattleZone-v4Image17.png", "BattleZoneImageFolder\BattleZone-v4Image31.png"])
# x.getAllClassesInOutput()

# classify_observation's init provides the outputs
# Format of parameters:
# (Path to Ilastik, Path to project file, path to export folder, path to inputs in arr form)

# getAllClassesInOutput records the different classes in integer form on the textfile "classifying.txt"