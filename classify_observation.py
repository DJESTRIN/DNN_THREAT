# Command should be python classify_observation.py
import subprocess
import os
import sys
import h5py
import numpy as np
from typing import List


class Classify_observation:
    """
    A class that represents an Ilastik project.

    :param pathToIlastik: A string representing the path to the Ilastik executable.
    :param projectFilePath: A string representing the path to the ilp project file.
    :param game: A string representing what game we are playing.
    """

    def __init__(self, pathToIlastik: str, projectFilePath: str, game: str):
        """
        Initializes a new instance of the Classify_observation class.

        :param pathToIlastik: A string representing the path to the Ilastik executable.
        :param projectFilePath: A string representing the path to the ilp project file.
        """
        self.pathToIlastik = pathToIlastik
        self.projectFilePath = projectFilePath
        self.current_dir = os.getcwd()
        self.game = game

    def exportClassesFromDir(self, exportFolder: str, inputDirPath: str):
        """
        Produces the export h5 files and places them in the exportFolder based on input files in inputDirPath.

        :param exportFolder: A string representing the path to the export folder where the h5 files will be stored.
                            If exportFolder doesn't exist, it will be created.
        :param inputDirPath: A string representing the path to the directory that contains the input files.
        :return: None
        """
        
        if not os.path.exists(exportFolder): # If the export folder doesn't exist, create it
            os.mkdir(exportFolder)
        
        inputFiles = [os.path.join(inputDirPath, f) for f in os.listdir(inputDirPath) if f.endswith('.png')] #only extracts .png files
        self.exportClassesFromFiles(exportFolder, inputFiles)

    def exportClassesFromFiles(self, exportFolder: str, inputFilePaths: List[str]) -> None:
        """
        Produces the export h5 files and places them in the exportFolder based on an array of inputFilePaths.

        :param inputFilePaths: A list of file paths representing the input files.
        :param exportFolder: A string representing the path to the export folder where the h5 files will be stored.
                            If exportFolder doesn't exist, it will be created.
        :return: None
        """ 
        
        if not os.path.exists(os.path.join(self.current_dir, exportFolder)): # If the export folder doesn't exist, create it
            os.mkdir(exportFolder)

        
        os.chdir(self.pathToIlastik) # operation must take place at the Ilastik file location
        inputFilesPathsArr = []
        for i in range(len(inputFilePaths)):
            inputFilesPathsArr.append(os.path.join(self.current_dir, inputFilePaths[i]))
        inputFilesPathsArr = ' '.join(inputFilesPathsArr)
        command = "ilastik.exe --headless --project={} --export_source=\"Simple Segmentation\" --output_filename_format={} {}".format(os.path.join(self.current_dir, self.projectFilePath), os.path.join(self.current_dir, exportFolder, '{nickname}_{result_type}.h5'), inputFilesPathsArr)

        if (len(command) > 8000): # Windows has a limit of about 8000 characters per command, so we divide the command in half if it is over
            half_len = len(inputFilePaths) // 2
            self.exportClassesFromFiles(exportFolder, inputFilePaths[:half_len])
            self.exportClassesFromFiles(exportFolder, inputFilePaths[half_len:])
        else:
            os.system(command)
            os.chdir(self.current_dir) # brings the os back to the original directory

    def displayAllClassesInOutputTextFile(self, exportFolder: str, exportTextFile: str, query: str ='all') -> None:
        """
        Displays what classes are in each output in a text file, depending on the query.

        :param exportFolder: A string representing the path to the folder that the function is pulling exports from.
        :param exportTextFile: A string representing the path to the text file that the function will write to.
        :param query: A string representing which classes we will be exporting. This should be either "all" or the name of a Class in Ilastik
        :return: None
        """
        os.chdir(self.current_dir)

        if not os.path.exists(os.path.dirname(exportTextFile)) and not os.path.dirname(exportTextFile) == "": # Create the directory if it does not exist
            os.makedirs(os.path.dirname(exportTextFile))

        if not os.path.isfile(exportTextFile): # Create the file if it does not exist
            with open(exportTextFile, "w") as file:
                pass

        with open(exportTextFile, "a") as textfile:
            textfile.write("\nQuery {}\n\n".format(query))
    
        exportFolderFiles = os.listdir(exportFolder)
        for file in exportFolderFiles:
            if file.endswith('.h5'):
                with h5py.File(os.path.join(exportFolder, file), 'r') as f:
                    for dataset_name in f: # The dataset is 'exported_data', but just in case that doesn't work we use this
                        dset = f[dataset_name]
                        nparr = np.array(dset).flatten()

                        values = []
                        for x in np.unique(nparr):
                            if query == 'all' or whichClass(self.game, x) == query:
                                values.append(whichClass(self.game, x))
                        if len(values) > 0:
                            values = ", ".join(values)
                        else:
                            values = "Not Found"

                        # for x in np.unique(nparr):
                        with open(exportTextFile, "a") as textfile:
                            textfile.write("{}: {}\n".format(file, values))
                os.remove(os.path.join(exportFolder, file))

def whichClass(game, num):
    if game == "SpaceInvaders":
        match(num):
            case 1:
                return "Agent"
            case 2:
                return "Enemy"
            case 3:
                return "Background"
            case 4:
                return "Bullet"
            case 5:
                return "Reward"
            case 6:
                return "Wall"
            case 7:
                return "Death"
            case 8:
                return "Boost"
    elif game == "BattleZone":
        match(num):            
            case 1:
                return "Agent"
            case 2:
                return "Enemy"
            case 3:
                return "Bullet"
            case 4:
                return "Death"
            case 5:
                return "Background"
            case 6:
                return "Boost"
# x = Classify_observation("C:\Program Files\ilastik-1.4.0", "BattleZoneVersion0.ilp", "BattleZone")
# x.exportClassesFromDir("Exports2", "TestingFolder")
# x.displayAllClassesInOutputTextFile("Exports2","testtext2.txt")
# x.displayAllClassesInOutputTextFile("Exports2",os.path.join("Exports3","testtext2.txt"))
# x.displayAllClassesInOutputTextFile("Exports",os.path.join("Exports4","testtext.txt"))