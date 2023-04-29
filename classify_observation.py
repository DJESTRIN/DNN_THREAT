import subprocess
import os
import sys

# Get the current working directory
current_dir = os.getcwd()

print("HEADS UP! This is assuming that you are using a Windows Computer and using Anaconda\n")
pathToIlastik = sys.argv[1]

os.chdir(pathToIlastik)

projectFile = sys.argv[2]

inputFiles = []

print(os.getcwd() + "\n")

for arg in sys.argv[3:]:
    inputFiles.append(arg)

for i in range(len(inputFiles)):
    inputFiles[i] = current_dir + '\\' + inputFiles[i]

inputFiles = ' '.join(inputFiles)

print("ilastik.exe --headless --project={}\{} {}".format(current_dir, projectFile, inputFiles))

os.system("ilastik.exe --headless --project={}\{} {}".format(current_dir, projectFile, inputFiles)) 