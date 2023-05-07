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
from itertools import product
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import shutil
import sys
from wordcloud import WordCloud

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
    SIMILAR_TRACKS = "#5cd0ed"

    POPULARITY = "#19b065"
    KEY = "#35606d"
    MODE = "#0e3d4b"
    DURATION = "#ed5c5e"
    LOUDNESS = "#570202"
    TEMPO = "#5f5f5f"
    TIME_SIGNATURE = "#272727"

    DANCEABILITY = "#853d9b"
    ENERGY = "#f58f2f"
    SPEECHINESS = "#15c7a0"
    ACOUSTICNESS = "#b48a52"
    INSTRUMENTALNESS = "#e489dc"
    LIVENESS = "#d63693"
    VALENCE = "#e0b010"

percentageColumns = [Columns.DANCEABILITY, Columns.ENERGY, Columns.SPEECHINESS, Columns.ACOUSTICNESS, Columns.INSTRUMENTALNESS, Columns.LIVENESS, Columns.VALENCE]
percentageColumnsAllCaps = ["DANCEABILITY", "ENERGY", "SPEECHINESS", "ACOUSTICNESS", "INSTRUMENTALNESS", "LIVENESS", "VALENCE"]
percentageColumnsAdjectives = ["danceable", "energetic", "talkative", "acoustic", "instrumental", "live", "positive"]


# Prepare temp and out dir

currentDir = os.getcwd()
tempDir = os.path.join(currentDir, "temp")
globalOutDir = os.path.join(currentDir, "out")

if os.path.exists(tempDir):
    shutil.rmtree(tempDir)

os.mkdir(tempDir)

if not os.path.exists(globalOutDir):
    os.mkdir(globalOutDir)

outDir = os.path.join(globalOutDir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
os.mkdir(outDir)





# copy and prepare csv file
if (len(sys.argv) != 2):
    print("syntax: spotify-playlist-analyzer <csv_file>")
    exit(1)

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
pdfNetworkPageSize = (math_utils.mmToInches(594), math_utils.mmToInches(594)) # A2 length
pdfWordCloudPageSize = (math_utils.mmToInches(297), math_utils.mmToInches(210)) # A4

def addTitle(title: str):
    """Append a title page to the pdf."""
    pdf_utils.addTitlePage(pdf, title, pdfPageSize)

playlistLength = dataFrame.shape[0]
pdf_utils.addFirstPage(pdf, pdfPageSize, playlistFileName, playlistLength)
print("Creating diagrams...")





# =======
# NETWORK
# =======
print("- network of similar tracks")
addTitle("Network of similar tracks")

# get all possible pairs of indexes
pairsDf = pd.DataFrame(product(dataFrame.index, dataFrame.index))

# remove duplicates (x, x), as it doesn't make sense to compare a track to itself
# remove pair "duplicates", as (x,y) and (y,x) is the same comparison

# removes all indexes of pairsDf (different from the original dataframe indexes)
# indicated in the provideed list. The provided list is the list of indexes where
# the condition is respected.
pairsDf = pairsDf.drop(pairsDf[pairsDf[0] >= pairsDf[1]].index)

# add metadata used for calculations: map each pair index to its value in the dataframe,
# in a column labeled as "label_name + index column" (0 or 1)

# ex: for Danceability on (0,2),
# Danceability0 = df[Danceability][0] (value at index 0)
# and Danceability1 = df[Danceability][2]
distanceColumns = []
for col in percentageColumns:
    for i in range(0,2):
        pairsDf[col+str(i)] = pairsDf[i].apply(lambda x: dataFrame[col][x])
    # calculate distance between both values
    distCol = col+"_Distance"
    distanceColumns.append(distCol)
    pairsDf[distCol] = abs(pairsDf[col+"0"] - pairsDf[col+"1"])

# calculate global distance for all pairs
GLOB_DIST_COL = "Global_Distance"
pairsDf[GLOB_DIST_COL] = pairsDf[distanceColumns].sum(axis=1) # horizontal sum instead of vertical

# keep only the smallest distances
#pairsDf = pairsDf.sort_values(GLOB_DIST_COL).head(playlistLength) # keep as many arcs as the number of tracks times x

# keep atleast one edge for each node
pairsDf = pairsDf.sort_values(GLOB_DIST_COL)

keptPairsDf = pd.DataFrame(columns=[0, 1, GLOB_DIST_COL])
edgesCountPerIndex = [0] * playlistLength # 0 for each index
MAX_EDGES_PER_INDEX = 2

for index, row in pairsDf.iterrows():
    i0 = int(row[0])
    i1 = int(row[1])
    count0 = edgesCountPerIndex[i0]
    count1 = edgesCountPerIndex[i1]
    if (count0 < MAX_EDGES_PER_INDEX or count1 < MAX_EDGES_PER_INDEX):
        # add if one of the index has no edge
        keptPairsDf.loc[len(keptPairsDf)] = [i0, i1, row[GLOB_DIST_COL]]
        edgesCountPerIndex[i0] += 1
        edgesCountPerIndex[i1] += 1

# build network
graph = nx.Graph()
# add vertices
graph.add_nodes_from(list(range(playlistLength)))
# add arcs
graph.add_edges_from(keptPairsDf[[0, 1]].to_numpy())
# draw
graphFig, graphAxes = plt.subplots()
graphFig.set_size_inches(pdfNetworkPageSize)
graphPos = nx.spring_layout(graph) # k (0 to 1) controls the distance between nodes, defaults to 0.1
edgeLabels = {}
for index, row in keptPairsDf.iterrows():
    edgeLabels[(row[0], row[1])] = round(row[GLOB_DIST_COL] * 100)

nx.draw(graph, graphPos, ax=graphAxes, with_labels=True, alpha=0.5, font_size=8, node_size=250, node_color=DataColors.SIMILAR_TRACKS)
# for bbox see
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/placing_text_boxes.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch
nx.draw_networkx_edge_labels(graph, graphPos, ax=graphAxes, edge_labels=edgeLabels, font_size=6, label_pos=0.35, bbox=dict(color='white', alpha = 0.15))
graphAxes.set_title("Network of similar tracks:\nclosest tracks (nodes) have an arc in between them")

pdf.savefig(graphFig)





# =================
# GENRES WORD CLOUD
# =================

print("- genre word cloud")
addTitle("Genres (approximation\nfrom artist genres)\nFull list available in a file")

artistGenresDf = dataFrame[[Columns.ARTIST_GENRES]]
genresList = {}
genresStr = ""
for index, row in artistGenresDf.iterrows():
    genresStr += f",{str(row[0])}".replace("nan","")
    ArtistGenresList = str(row[0]).split(",")
    for g in ArtistGenresList:
        if (g != "nan"):
            if g in genresList:
                genresList[g] += 1
            else:
                genresList[g] = 1

# iloc reverses the order to have values in descending order.
genresDf = pd.DataFrame(genresList.items(), columns=["Genre", "Count"]).sort_values("Count").iloc[::-1]

# CSV
genresDf.index.name = "index"
genresDf.to_csv(os.path.join(outDir, "genres.csv"))

# word cloud - genres
wordcloud = WordCloud(width=900,height=500,background_color='white').generate_from_frequencies(frequencies=genresList)
genresFig, genresAxes = plt.subplots() #1 row, 1 col
genresFig.set_size_inches(pdfWordCloudPageSize)
genresAxes.imshow(wordcloud, interpolation="bilinear")
genresAxes.set_axis_off()
pdf.savefig(genresFig)
plt.close(genresFig)

# word cloud - genres words
wordcloud = WordCloud(width=900,height=500,background_color='white').generate_from_text(genresStr)
genreWordsFig, genreWordsAxes = plt.subplots() #1 row, 1 col
genreWordsFig.set_size_inches(pdfWordCloudPageSize)
genreWordsAxes.imshow(wordcloud, interpolation="bilinear")
genreWordsAxes.set_axis_off()
pdf.savefig(genreWordsFig)
plt.close(genreWordsFig)





# ===========
# RADAR CHART
# ===========

print("- radar chart")
addTitle("Average playlist profile")

def buildRadarChart():
    """
    Creates a radar / spider chart of the average value of multiple columns
    based on: https://www.pythoncharts.com/matplotlib/radar-charts/
    """

    # prepare data
    columnNames = percentageColumns.copy()
    # labels = columnNames.copy() # Technically unused...
    labels = []
    # see https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features
    # ORDER IS IMPORTANT
    labels.append("Danceability (suitable for dancing)")
    labels.append("Energy (intensity and activity)")
    labels.append("Speechiness\n(spoken words,\n>0.66 => almost\nonly words (ex: podcast),\n<0.33 => almost\nonly music)")
    labels.append("Acousticness (not electronic instruments)")
    labels.append("Instrumentalness (no vocals)")
    labels.append("Liveness (performed live)")
    labels.append("Valence (positivity, happiness)")
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
    plt.close(radarFig)

buildRadarChart()





# ============
# DISTRIBUTION
# ============

print("- distribution")
addTitle("Distribution")

def buildDistribution(name: str, distribDataColumn: str, rMin: int, rMax: int, rStep: int, scale: int=1, scaleOnlyDisplay: bool=True, unit: str="", xLabel: str="", color: str="C0"):
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
    - `scaleOnlyDisplay`: if false, data is scaled before being processed. If true, only the displayed text is altered. It is useful for ranges such as [0;1] to be usable.
    - `unit`: String to display a unit next to the axis label.
    - `xlabel`: override the label for the x axis with a custom one, instead of the column name.
    - `color`: Sets the bars color
    """

    print(".", end="", flush=True)

    distribDf = dataFrame[[distribDataColumn]].copy()
    if not scaleOnlyDisplay:
        distribDf[distribDataColumn] *= scale

    # Bins
    distribBins = np.arange(rMin, rMax+1, rStep)
    # add min and max values from data to bins,
    # to show values below min and above max with bars on the sides of the chart
    maxBin = max(rMax, math.ceil(distribDf.max())) # if the order is broken because (data_max < rMax), hist() will fail
    minBin = min(rMin, math.floor(distribDf.min()))
    distribBins = np.append(distribBins, maxBin) # max bin to the end
    distribBins = np.insert(distribBins, 0, minBin) # min bin to the start

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
    xTicks = distribBins[1:-1] # exclude min and max added above, they should not be displayed.
    if scaleOnlyDisplay:
        xTicks = xTicks * scale
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
    plt.close(distribFig)


# Distributions

# Popularity
buildDistribution("How popular are the tracks of your playlist?\n(0 are niche, 100 are most streamed tracks)",
                  Columns.POPULARITY,
                  rMin=0, rMax=100, rStep=5,
                  color=DataColors.POPULARITY)

# Duration
buildDistribution("How long are the tracks?",
                  Columns.DURATION,
                  rMin=60_000, rMax=60_000*10, rStep=30_000,
                  scale=(1/60000),
                  xLabel="Duration",
                  unit="minutes",
                  color=DataColors.DURATION)

# Key
buildDistribution(f"Key distribution (0 (fist bar) is C, 1 is C#, etc. and -1 if unknown).\nFor such histograms (1 bar per unit), read the value to the left of the bar.",
                  Columns.KEY,
                  rMin=0, rMax=12, rStep=1,
                  color=DataColors.KEY)

# Mode
buildDistribution(f"Mode distribution (1 is major, 0 is minor)",
                  Columns.MODE,
                  rMin=0, rMax=2, rStep=1,
                  color=DataColors.MODE)

# Loudness
buildDistribution(f"How loud are the tracks?",
                  Columns.LOUDNESS,
                  rMin=-60, rMax=0, rStep=5,
                  unit="dB",
                  color=DataColors.LOUDNESS)

# Tempo
buildDistribution(f"How fast are the tracks?",
                  Columns.TEMPO,
                  rMin=50, rMax=200, rStep=10,
                  color=DataColors.TEMPO)

# Time Signature
buildDistribution(f"What is the estimated time signature of the tracks?\n(3/4 to 7/4, value to the left of the bar)",
                  Columns.TIME_SIGNATURE,
                  rMin=3, rMax=7, rStep=1,
                  color=DataColors.TIME_SIGNATURE)

# Percentages
for percentage in percentageColumnsAllCaps:
    buildDistribution(f"Amount of {getattr(Columns, percentage)} in the tracks.\n See the profile radar chart for the meaning.",
                      getattr(Columns, percentage),
                      rMin=0, rMax=100, rStep=5,
                      scale=100,
                      scaleOnlyDisplay=False,
                      color=getattr(DataColors, percentage))
    
print()





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

    print(".", end="", flush=True)

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
                    distribBins1[i] + config1.rStep/2,
                    distribBins2[j] + config2.rStep/2,
                    int(hist.T[j,i]), #T: transposed array
                    color="white", ha="center", va="center", fontweight="bold", fontsize=16)
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground="#00000077")])

    pdf.savefig(crossedDistribFig)
    plt.close(crossedDistribFig)


# Crossed distributions

# Percentages
crossedPairsDone = []
for percentageA in percentageColumns:
    for percentageB in percentageColumns:
        # done ?
        done = False
        for pair in crossedPairsDone:
            if ((pair[0] == percentageA and pair[1] == percentageB) or (pair[0] == percentageB and pair[1] == percentageA)):
                done = True
                break
        
        if not (percentageA == percentageB) and not done:
            crossedPairsDone.append((percentageA, percentageB))
            buildCrossedDistribution(f"{percentageA} vs {percentageB}",
                    CrossedDistribConfig(percentageA, rMin=0, rMax=1, rStep=0.1),
                    CrossedDistribConfig(percentageB, rMin=0, rMax=1, rStep=0.1))

# Key and mode
buildCrossedDistribution(f"Key vs Mode",
        CrossedDistribConfig(Columns.KEY, rMin=0, rMax=12, rStep=1),
        CrossedDistribConfig(Columns.MODE, rMin=0, rMax=2, rStep=1))

print()





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
    
    print(".", end="", flush=True)

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
    if (max(rankValues) < 0): # invert if negative to stay consistent
        rankAxes.invert_xaxis()
    rankAxes.bar_label(barContainer, padding=2)
    rankAxes.set_title(name)
    label = sortingColumn if (xLabel == "") else xLabel
    if (not unit == ""):
        label += f" ({unit})"
    rankAxes.set_xlabel(label)

    pdf.savefig(rankFig)
    plt.close(rankFig)


def buildSimilarTracksRanking(name: str, maxRows: int=20, least: bool=False, scale: int=1, xLabel: str="", color: str="C0"):
    """
    Builds a ranking of similar tracks

    Parameters:
    - `name`: name of the figure.
    - `sortingColumn`: the dataFrame column to use in `dataFrame`.
    - `maxRows`: The maximum of tracks displayed in the figure.
    - `least`: Sort by lowest values instead of highest values.
    - `scale`: can be used to apply a factor to the ranked column (on a dataframe copy).
    - `xlabel`: override the label for the x axis with a custom one, instead of the column name.
    - `color`: Sets the bars color
    """
    # setup and style
    rankFig, rankAxes = plt.subplots()
    rankFig.set_size_inches(pdfPageSize)
    rankFig.subplots_adjust(left=0.50)

    # get ranking
    sortingColumn = GLOB_DIST_COL
    rankDf = pairsDf[[0, 1, sortingColumn]].copy()
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
    for index, row in rankDf.iterrows():
        label = f"{row[0]} and {row[1]}"
        rankLabels.append(label)
    rankValues = rankDf[sortingColumn].to_numpy()

    # create plot and append information
    barContainer = rankAxes.barh(rankLabels, rankValues, color=color) # inverts order of dataframe on display
    rankAxes.bar_label(barContainer, padding=2)
    rankAxes.set_title(name)
    label = sortingColumn if (xLabel == "") else xLabel
    rankAxes.set_xlabel(label)

    pdf.savefig(rankFig)
    plt.close(rankFig)



# Rankings

# Similar tracks
buildSimilarTracksRanking("Most similar tracks", least=True, color=DataColors.SIMILAR_TRACKS)
buildSimilarTracksRanking("Least similar tracks", color=DataColors.SIMILAR_TRACKS)

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

# Basic ranking
rankingColumns = ["POPULARITY", "LOUDNESS", "TEMPO"] + percentageColumnsAllCaps
rankingAdjectives = ["popular", "loud", "fast"] + percentageColumnsAdjectives
rankingUnits = ["", "dB", "BPM"]
rankingScales = [1, 1, 1]
for percentage in percentageColumnsAllCaps:
    rankingUnits.append("%")
    rankingScales.append(100)

for i in range(len(rankingColumns)):
    adj = rankingAdjectives[i]
    column = rankingColumns[i]

    buildRanking(f"Most {adj} tracks", getattr(Columns, column),
                 color=getattr(DataColors, column),
                 unit=rankingUnits[i],
                 scale=rankingScales[i])
    
    buildRanking(f"Least {adj} tracks", getattr(Columns, column),
                 color=getattr(DataColors, column),
                 least=True,
                 unit=rankingUnits[i],
                 scale=rankingScales[i])

print()





# ==============================================================================================================
# ==============================================================================================================
# ==============================================================================================================

# export pdf
print("saving PDF...")
pdf.close()

print("done!")

# TODO
# - folder name with playlist name
# - more data