# MIT License

# Copyright (c) 2023 Picorims

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dependencies
from datetime import datetime
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import sys

# Other project modules
import math_utils
import pdf_utils

class Columns():
    """
    Enumeration of relevant CSV columns
    """
    TRACK_NAME = "Track Name"
    ARTIST_NAMES = "Artist Name(s)"
    ALBUM_NAME = "Album Name"
    ARTIST_GENRES = "Artist Genres"

    POPULARITY = "Popularity"
    KEY = "Key"
    DURATION = "Track Duration (ms)"
    LOUDNESS = "Loudness"
    TEMPO = "Tempo"
    TIME_SIGNATURE = "Time Signature"

    DANCEABILITY = "Danceability"
    ENERGY = "Energy"
    SPEECHINESS = "Speechiness"
    ACOUSTICNESS = "Acousticness"
    INSTRUMENTALNESS = "Instrumentalness"
    LIVENESS = "Liveness"
    VALENCE = "Valence"
    


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

dataFrame = pd.read_csv(csvPath, header=0)





# Build index file

print("Building index...")
indexDf = dataFrame[[Columns.TRACK_NAME, Columns.ARTIST_NAMES, Columns.ALBUM_NAME]].copy()
indexDf.index.name = "index"
indexDf.to_csv(os.path.join(outDir, "index.csv"))





















# DIAGRAMS ==============================================================================================================
# DIAGRAMS ==============================================================================================================
# DIAGRAMS ==============================================================================================================

plt.style.use("seaborn-v0_8-whitegrid")
print("Setting up PDF...")

# Setup PDF

playlistFileNoExtension = os.path.splitext(sys.argv[1])[0]
playlistFileName = os.path.split(playlistFileNoExtension)[1]
pdfName = "stats_playlist_" + playlistFileName + ".pdf"
pdf = PdfPages(os.path.join(outDir, pdfName))
pdfPageSize = (math_utils.mmToInches(210), math_utils.mmToInches(148)) # A5

def addTitle(title: str):
    """Append a title page to the pdf."""
    pdf_utils.addTitlePage(pdf, title, pdfPageSize)

print("Creating diagrams...")




# ==========
# POPULARITY (between 0 and 100)
# ==========

print("- popularity frequency")
addTitle("Popularity")

# LEGACY WAY OF CALCULATING FREQUENCIES USING PANDAS
# # Round all values by multiple of 5 to calculate frequency by bins of 5 (0-4 ; 5-9; 10-14; ... ; 95-100)
# popularitiesByRange = dataFrame[Columns.POPULARITY].apply(lambda x : math_utils.floorByStepsOf(5, x))
# # Handle 100
# popularitiesByRange = popularitiesByRange.replace(100, 95)

# # Count every bin and store it in a dictiorary
# popularityFreqs = popularitiesByRange.value_counts().sort_index()

# Bins
popBins = list(range(0, 100, 5))

popFig, popAxes = plt.subplots() #1 row, 1 col
popFig.set_size_inches(pdfPageSize)
counts, edges, bars = popAxes.hist(dataFrame[Columns.POPULARITY].to_numpy(), bins=popBins)
plt.bar_label(bars)
popAxes.set_title("How popular are the tracks of your playlist ? (between 0 and 100)")
popAxes.set_xlabel("Popularity")
popAxes.set_ylabel("Number of occurences")
popAxes.set_xticks(popBins) # force display tick at every step of 5

pdf.savefig(popFig)





# ========
# RANKINGS (between 0 and 100)
# ========

addTitle("Rankings")

# ranking by popularity
# ---------------------

print("- rankings")
# setup and style
mostPopRankFig, mostPopRankAxes = plt.subplots()
mostPopRankFig.set_size_inches(pdfPageSize)
mostPopRankFig.subplots_adjust(left=0.50)

# get ranking
mostPopRankDf = dataFrame[[Columns.TRACK_NAME, Columns.POPULARITY]]
mostPopRankDf = mostPopRankDf.sort_values(Columns.POPULARITY).tail(20)

# build list of labels and values
mostPopRankLabels = []
for index, row in mostPopRankDf.iterrows():
    mostPopRankLabels.append(f"{index} - {row[Columns.TRACK_NAME]}")
mostPopRankValues = mostPopRankDf[Columns.POPULARITY].to_numpy()

# plot and labels
barContainer = mostPopRankAxes.barh(mostPopRankLabels, mostPopRankValues) # inverts order of dataframe on display
mostPopRankAxes.bar_label(barContainer, padding=2)
mostPopRankAxes.set_title("Most popular tracks")
mostPopRankAxes.set_xlabel("Popularity")

pdf.savefig(mostPopRankFig)






# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

# export pdf
print("saving PDF...")
pdf.close()

print("done!")