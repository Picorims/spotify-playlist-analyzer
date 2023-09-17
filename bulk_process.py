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

import os
import subprocess
import sys




if (len(sys.argv) != 2):
    print("syntax: bulk_process.py <directory>")
    exit(1)

if not os.path.exists(sys.argv[1]):
    print("directory does not exist.")
    exit(1)

csvList = os.listdir(sys.argv[1])

for entry in csvList:
    path = os.path.join(sys.argv[1], entry)
    extension = os.path.splitext(path)[1]

    if os.path.isfile(path) and extension == ".csv":
        print("=== processing: ", entry)
        subprocess.call(f"py spotify_playlist_analyzer.py {path}", shell=True)
    else:
        print("=== ignoring: ", entry)