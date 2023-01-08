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
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
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
    MODE = "Mode" #0 is minor, 1 is majors
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
    
class DataColors:
    POPULARITY = "#19b065"
    DURATION = "#ed5c5e"


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





# ========
# RANKINGS
# ========

print("- radar chart")
addTitle("Average playlist profile")

def buildRadarChart():
    """
    Creates a radar / spider chart of the average value of multiple columns
    based on: https://www.pythoncharts.com/matplotlib/radar-charts/
    """

    # prepare data
    columnNames = [Columns.DANCEABILITY, Columns.ENERGY, Columns.SPEECHINESS, Columns.ACOUSTICNESS, Columns.INSTRUMENTALNESS, Columns.LIVENESS, Columns.VALENCE]
    labels = columnNames.copy()
    labels[5] = "Liveness (performed live)"
    labels[6] = "Valence (positivity, happiness)"
    averages = []
    for col in columnNames:
        averages.append(dataFrame[col].mean())
    nbValues = len(columnNames)

    # split the circle into angles for each column
    # false = exclude stop, tolist() to be able to append values
    angles = np.linspace(0, 2*np.pi, nbValues, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    averages.append(averages[0])
    angles.append(angles[0])

    # figure
    radarFig, radarAxes = plt.subplots(subplot_kw=dict(polar=True))
    radarFig.set_size_inches(pdfPageSize)
    radarAxes.set_title("Average / mean value for different parameters")
    # max value
    radarAxes.set_ylim(0,1)
    # or ax.set_rgrids([20, 40, 60, 80, 100])
    # line
    radarAxes.plot(angles, averages, linewidth=1)
    # fill
    radarAxes.fill(angles, averages, alpha=0.25)

    # Fix axis to go in the right order and start at 12 o'clock.
    radarAxes.set_theta_offset(np.pi / 2) # rotate
    radarAxes.set_theta_direction(-1) # reverse order

    # Draw axis lines for each angle and label (and add the labels as well).
    radarAxes.set_thetagrids(np.degrees(angles)[0:-1], labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(radarAxes.get_xticklabels(), angles):
        if angle in (0, np.pi): # equals 0 or pi
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Set position of y-labels (0-100) to be in the middle
    # of the first two axes.
    radarAxes.set_rlabel_position(180 / nbValues)

    pdf.savefig(radarFig)

buildRadarChart()





# ============
# DISTRIBUTION
# ============

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
    - `rMin`, `rMax`, `rStep`: range that defines the bins for the histogram (min, max (included!), and step for slicing the whole range in bins).
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
    distribAxes.set_xlabel(finalXLabel)
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
        rMin=0, rMax=100, rStep=5, color=DataColors.POPULARITY)

# Duration
buildDistribution("How long are the tracks?\n",
        Columns.DURATION,
        rMin=60_000, rMax=60_000*10, rStep=30_000,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color=DataColors.DURATION)





# ========
# RANKINGS
# ========

print("- crossed distribution")
addTitle("Crossed distribution")

class CrossedDistribConfig:
    """
    class that holds all the data for configuring one axis of the 2d histogram,
    to make data transmission easier.
    """
    distribDataColumn: str
    rMin: int
    rMax: int
    rStep: int
    scale: int=1
    unit: str=""
    label: str=""

    def __init__(self, distribDataColumn: str, rMin: int, rMax: int, rStep: int, scale: int=1, unit: str="", label: str="") -> None:
        self.distribDataColumn = distribDataColumn
        self.rMin = rMin
        self.rMax = rMax
        self.rStep = rStep
        self.scale = scale
        self.unit = unit
        self.label = label


def buildCrossedDistribution(name: str, config1: CrossedDistribConfig, config2: CrossedDistribConfig):
    """
    Builds a 2d histogram based on the config. Concentration of values in an area
    is indicated by colors.

    parameters:
    - `config1`, `config2`: configuration for each axis of the 2d histogram.

    config parameters:
    - `distribDataColumn`: the dataFrame column to use in `dataFrame`.
    - `rMin`, `rMax`, `rStep`: range that defines the bins for the histogram (min, max (included!), and step for slicing the whole range in bins).
    Note that out of range values are ignored.
    - `scale`: can be used to apply a factor to the ranked column (on a dataframe copy).
    - `unit`: String to display a unit next to the axis label.
    - `label`: override the label for the x axis with a custom one, instead of the column name.
    """

    crossedDistribDf = dataFrame[[config1.distribDataColumn, config2.distribDataColumn]]

    # Bins
    distribBins1 = np.arange(config1.rMin, config1.rMax + config1.rStep, config1.rStep) # + step ensures max (and not more) is included
    distribBins2 = np.arange(config2.rMin, config2.rMax + config2.rStep, config2.rStep)

    # figure and plotting
    crossedDistribFig, crossedDistribAxes = plt.subplots() #1 row, 1 col
    crossedDistribFig.set_size_inches(pdfPageSize)
    hist, xBins, yBins, image = crossedDistribAxes.hist2d(
            crossedDistribDf[config1.distribDataColumn].to_numpy(),
            crossedDistribDf[config2.distribDataColumn].to_numpy(),
            bins=(distribBins1, distribBins2),
            cmap="viridis")
    crossedDistribFig.colorbar(image)
    
    # additional information
    crossedDistribAxes.set_title(name)

    finalXLabel = config1.distribDataColumn if (config1.label == "") else config1.label
    if (not config1.unit == ""):
        finalXLabel += f" ({config1.unit})"
    crossedDistribAxes.set_xlabel(finalXLabel)
    
    finalYLabel = config2.distribDataColumn if (config2.label == "") else config2.label
    if (not config2.unit == ""):
        finalYLabel += f" ({config2.unit})"
    crossedDistribAxes.set_ylabel(finalYLabel)

    # tweak axis
    crossedDistribAxes.set_xticks(distribBins1)
    crossedDistribAxes.set_yticks(distribBins2)
    crossedDistribAxes.set_xlim(config1.rMin, config1.rMax)
    crossedDistribAxes.set_ylim(config2.rMin, config2.rMax)

    # print values in all bins
    # https://stackoverflow.com/questions/43538581/printing-value-in-each-bin-in-hist2d-matplotlib
    for i in range(len(distribBins1)-1):
        for j in range(len(distribBins2)-1):
            text = crossedDistribAxes.text(
                    distribBins1[j] + config1.rStep/2,
                    distribBins2[i] + config2.rStep/2,
                    int(hist.T[i,j]), #T: transposed array
                    color="white", ha="center", va="center", fontweight="bold", fontsize=16)
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground="#00000077")])

    pdf.savefig(crossedDistribFig)


# Crossed distributions

buildCrossedDistribution("Danceability vs Energy",
        CrossedDistribConfig(Columns.DANCEABILITY, rMin=0, rMax=1, rStep=0.1),
        CrossedDistribConfig(Columns.ENERGY, rMin=0, rMax=1, rStep=0.1))





# ========
# RANKINGS
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
buildRanking("Most popular tracks", Columns.POPULARITY, color=DataColors.POPULARITY)
buildRanking("Least popular tracks", Columns.POPULARITY, least=True, color=DataColors.POPULARITY)

#Duration
buildRanking("Longest tracks",
        Columns.DURATION,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color=DataColors.DURATION)
buildRanking("Shortest tracks",
        Columns.DURATION,
        least=True,
        scale=(1/60000),
        xLabel="Duration",
        unit="minutes",
        color=DataColors.DURATION)






# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

# export pdf
print("saving PDF...")
pdf.close()

print("done!")

# TODO
# - folder name with playlist name
# - graph
# - more data