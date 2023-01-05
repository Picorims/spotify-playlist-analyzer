# spotify-playlist-analyzer
Python script that generates all kinds of visualization based on a spotify playlist exported via Exportify

## Setup (Windows)

**for Unix/macOS commands visit https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment**

- Make sure Python 3.11 is installed on your computer (this is the tested version).
- Setup a virtual environment:
```
py -m venv env
```

- Activate the environment:
```
.\env\Scripts\activate
```

- Install dependencies:
```
py -m pip install -r requirements.txt
```

- Deactivate the environment:
```
deactivate
```

## Usage

- Get the playlist metadata from https://watsonbox.github.io/exportify/ (make sure to enable optional data in the settings!)
- Enable the environment in your terminal
- use the following command
```
py spotify_playlist_analyzer.py <path_to_csv>
```
- The output is in the `out` folder of the project, in a folder named after the current time.

## Updating dependencies (for development purposes only)

- Use pip to update dependencies
- Update requirements.txt:
```
py -m pip freeze | Set-Content requirements.txt
```

## License

MIT License