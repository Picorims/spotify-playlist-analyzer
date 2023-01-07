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

from typing import Tuple
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def addTitlePage(pdf: PdfPages, title: str, pageSize: Tuple[float,float]):
    """
    Creates a new empty figure with the provided title, and append it to the pdf.
    Inspired from: https://stackoverflow.com/questions/49444008/add-text-with-pdfpages-matplotlib
    """
    fig = plt.figure(figsize=pageSize)
    fig.clear()
    fig.text(0.5, 0.5, title, size=48, ha="center")
    pdf.savefig(fig)



def addFirstPage(pdf: PdfPages, pageSize: Tuple[float,float], playlistName: str, nbTracks: int):
    """
    Add a presentation page with the playlist name, and other information
    """
    fig = plt.figure(figsize=pageSize)
    fig.clear()
    fig.text(0.5, 0.5, "Playlist statistics for:\n" + playlistName, size=48, ha="center")
    fig.text(0.5, 0.25, f"analyzing {str(nbTracks)} tracks", size=24, ha="center")
    fig.text(0.5, 0.05, "Made using Spotify Playlist Analyzer, by Picorims.\nThis tool is not affiliated to Spotify.", size=12, ha="center")
    pdf.savefig(fig)
