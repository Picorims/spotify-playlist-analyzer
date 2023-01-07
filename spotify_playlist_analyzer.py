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
import math
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

playlistLength = dataFrame.shape[0]
pdf_utils.addFirstPage(pdf, pdfPageSize, playlistFileName, playlistLength)
print("Creating diagrams...")




# ==========
# POPULARITY (between 0 and 100)
# ==========

print("- distribution")
addTitle("Distribution")

def buildDistribution(name: str, distribDataColumn: str, rMin: int, rMax: int, rStep: int, scale: int=1, unit: str="", xLabel: str="", color: str="C0"):
    """
    Builds an histogram with the given name,
    displaying the distribution of values for the given column.
    Labels contain the number of occurences, and ticks contain
    the ranges.
    
    Parameters:
    - `name`: name of the figure.
    - `distribDataColumn`: the dataFrame column to use in `dataFrame`.
    - `min`, `max`, `step`: range that defines the bins for the histogram (min, max (included!), and step for slicing the whole range in bins).
    Note that bars will be added for all values below min and above max. 
    - `scale`: can be used to apply a factor to the ranked column (on a dataframe copy).
    - `unit`: String to display a unit next to the axis label.
    - `xlabel`: override the label for the x axis with a custom one, instead of the column name.
    - `color`: Sets the bars color
    """

    distribDf = dataFrame[distribDataColumn]

    # Bins
    distribBins = np.arange(rMin, rMax+1, rStep)
    # add min and max values from data to bins,
    # to show values below min and above max with bars on the sides of the chart
    maxBin = max(rMax, math.ceil(distribDf.max())) # if the order is broken because (data_max < rMax), hist() will fail
    minBin = min(rMin, math.floor(distribDf.min()))
    distribBins = np.append(distribBins, maxBin)
    distribBins = np.insert(distribBins, 0, minBin)

    # figure and plotting
    distribFig, distribAxes = plt.subplots() #1 row, 1 col
    distribFig.set_size_inches(pdfPageSize)
    counts, edges, patches = distribAxes.hist(distribDf.to_numpy(), bins=distribBins, color=color)

    # bar labels
    # plt.bar_label(bars) (bars being the 3rd return value of hist()) can't be used.
    # Bars on the side are very wide due to going all the way down/up to min/max.
    # To make them loop equally sized, the chart is cropped on the side.
    # However labels are still centered by bar_label, and thus the labels for the first
    # and last bars are out of bound. Thus, it has to be done manually
    spacing = 0.01 * max(counts)
    # the first bar label is calculated from the right, not the left, as the left is far beyond the chart boundaries.
    distribAxes.text(distribBins[1] - rStep/2, counts[0] + spacing, int(counts[0]), ha="center") # x, y, value
    for i in range(1, len(counts)):
        distribAxes.text(distribBins[i] + rStep/2, counts[i] + spacing, int(counts[i]), ha="center")
    
    # additional information
    distribAxes.set_title(name)
    finalXLabel = distribDataColumn if (xLabel == "") else xLabel
    if (not unit == ""):
        finalXLabel += f" ({unit})"
    distribAxes.set_xlabel(distribDataColumn)
    distribAxes.set_ylabel("Number of occurences")
    distribAxes.set_xlim(rMin - rStep, rMax + rStep)

    # tweak x axis ticks
    # scaling first (scaling the ticks is sufficient for the chart to remain accurate, and is simpler)
    xTicks = distribBins[1:-1] * scale # exclude min and max added above, they should not be displayed.
    xTicks = xTicks.astype(str) 
    # then adding text for indicating above n and below n more explicitely
    # xTicks[0] = f"<{xTicks[0]} | {xTicks[0]}"
    # xTicks[-1] = f"{xTicks[-1]} | >{xTicks[-1]}"
    distribAxes.text(distribBins[1] - rStep/2, -8 * spacing, "below", ha="center")
    distribAxes.text(distribBins[-2] + rStep/2, -8 * spacing, "above", ha="center")
    # and finally modifying ticks
    distribAxes.set_xticks(distribBins[1:-1]) # Create FixedLocator, otherwise set_xticklabels raise a warning and ignore new labels
    distribAxes.set_xticklabels(xTicks) # force display tick at every step

    # see also: https://stackoverflow.com/questions/26218704/matplotlib-histogram-with-collection-bin-for-high-values

    pdf.savefig(distribFig)


# Distributions

# Popularity
buildDistribution("How popular are the tracks of your playlist?\n(0 are niche, 100 are most streamed tracks)",
        Columns.POPULARITY,
        rMin=0, rMax=100, rStep=5)

# Duration
buildDistribution("How long are the tracks?\n",
        Columns.DURATION,
        rMin=60_000, rMax=60_000*10, rStep=30_000,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color="#e83c3f")





# ========
# RANKINGS (between 0 and 100)
# ========

print("- rankings")
addTitle("Rankings")

def buildRanking(name: str, sortingColumn: str, maxRows: int=20, least: bool=False, scale: int=1, unit: str="", xLabel: str="", color: str="C0"):
    """
    Builds an horizontal bar chart with the given name,
    displaying a list of highest or lowest values of a given column.
    Labels contain the index and name of the selected tracks.
    
    Parameters:
    - `name`: name of the figure.
    - `sortingColumn`: the dataFrame column to use in `dataFrame`.
    - `maxRows`: The maximum of tracks displayed in the figure.
    - `least`: Sort by lowest values instead of highest values.
    - `scale`: can be used to apply a factor to the ranked column (on a dataframe copy).
    - `unit`: String to display a unit next to the axis label.
    - `xlabel`: override the label for the x axis with a custom one, instead of the column name.
    - `color`: Sets the bars color
    """
    # setup and style
    rankFig, rankAxes = plt.subplots()
    rankFig.set_size_inches(pdfPageSize)
    rankFig.subplots_adjust(left=0.50)

    # get ranking
    rankDf = dataFrame[[Columns.TRACK_NAME, sortingColumn]].copy()
    # scaling
    rankDf[sortingColumn] *= scale
    if (not least):
        # most
        rankDf = rankDf.sort_values(sortingColumn).tail(maxRows)
    else:
        # least
        #[::-1] reverses the list, see https://www.codingem.com/reverse-slicing-in-python/
        rankDf = rankDf.sort_values(sortingColumn).head(maxRows).iloc[::-1]

    # build list of labels and values
    rankLabels = []
    STR_MAX_LEN = 50
    for index, row in rankDf.iterrows():
        label = f"{index} - {row[Columns.TRACK_NAME]}"
        if len(label) > STR_MAX_LEN:
            label = label[0:STR_MAX_LEN] + "..."
        rankLabels.append(label)
    rankValues = rankDf[sortingColumn].to_numpy()

    # create plot and append information
    barContainer = rankAxes.barh(rankLabels, rankValues, color=color) # inverts order of dataframe on display
    rankAxes.bar_label(barContainer, padding=2)
    rankAxes.set_title(name)
    label = sortingColumn if (xLabel == "") else xLabel
    if (not unit == ""):
        label += f" ({unit})"
    rankAxes.set_xlabel(label)

    pdf.savefig(rankFig)


# Rankings

# Popularity
buildRanking("Most popular tracks", Columns.POPULARITY)
buildRanking("Least popular tracks", Columns.POPULARITY, least=True)

#Duration
buildRanking("Longest tracks",
        Columns.DURATION,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color="#e83c3f")
buildRanking("Shortest tracks",
        Columns.DURATION,
        least=True,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color="#e83c3f")






# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

# export pdf
print("saving PDF...")
pdf.close()

print("done!")