from datetime import datetime
import numpy as np
import os
import pandas as pd
import shutil
import sys

class Columns():
    TRACK_NAME = "Track Name"
    ARTIST_NAMES = "Artist Name(s)"
    ALBUM_NAME = "Album Name"

# Prepare temp and out dir

currentDir = os.getcwd()
tempDir = os.path.join(currentDir, "temp")
globalOutDir = os.path.join(currentDir, "out")

if os.path.exists(tempDir):
    shutil.rmtree(tempDir)

os.mkdir(tempDir)

if not os.path.exists(globalOutDir):
    os.mkdir(globalOutDir)

outDir = os.path.join(globalOutDir, datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f"))
os.mkdir(outDir)





# copy and prepare csv file

print("Preparing CSV loading...")
csvPath = os.path.join(tempDir, "data.csv")
shutil.copyfile(sys.argv[1], csvPath)

# csvFile = open(csvPath, "r", encoding="utf-8")
# csvLinesR = csvFile.readlines()
# csvFile.close()
# csvFile = open(csvPath, "w", encoding="utf-8")
# csvLinesW = []

# for line in csvLinesR:
#     csvLinesW.append(line.replace('","','"|"'))

# csvFile.writelines(csvLinesW)
# csvFile.close()





# Load CSV LEGACY

# Note for numpy for the future:
# - do not handle string. example: "a,b","c" ==> ["a, b", "c"]. A separator not used anywhere is required.
# - consider a '#' as a comment character, which means it ignores content after a # on a line.

# genfromtxt handle empty values, using the NULL utf-8 character to disable comments
# data = np.genfromtxt(csvPath, delimiter="|", dtype=str, encoding="utf-8", comments="�")





# Load CSV

data = pd.read_csv(csvPath, header=0)





# Build index file

print("Building index...")
indexDf = data[[Columns.TRACK_NAME, Columns.ARTIST_NAMES, Columns.ALBUM_NAME]].copy()
indexDf.index.name = "index"
indexDf.to_csv(os.path.join(outDir, "index.csv"))





# DIAGRAMS

