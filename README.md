# Chart Race

## Introduction

Chart Race is a Python package that allows you to create animated chart races. It provides functions to generate video frames, and create chart race videos. The package also includes a command-line interface for easy usage.

### Video example

<p>
    <img width="700" src="https://user-images.githubusercontent.com/8163093/205472725-677cab85-64a6-4e92-b173-5930037c8d90.gif" />
</p>

## Pre-requisites

- Python 3.8 or higher

## Installation

To install Chart Race, you can clone the repository from GitHub:

```
git clone https://github.com/Igorxp5/chart-race.git
```

## Usage

To use Chart Race, you can import the necessary functions from the package and call them in your code. Here are some of the main functions available in the Python API:

## Command Line Interface

Chart Race also provides a command-line interface for easy usage. Here is an example of how to use it:

```
chart-race -t "My Chart Race" -l en_US entities.csv entries.csv output.mp4
```

### Arguments

- `entities_file`: CSV filepath with all entities. The columns must be: key, label, image_path
- `entries_files`: CSV filepath with all entries. The columns must be: entity_key, timestamp
- `output_file`: Output file path for the video generated

### Options

- `-t, --title TITLE`: Title of the chart race (default: CHART RACE)
- `-s, --video-speed VIDEO_SPEED`: Video speed rate (drop frame) (default: 1)
- `-l, --locale LOCALE`: Language for the date (default: en_US)
- `-g, --gray-out`: Gray out entities who do not have entries anymore (default: False)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
