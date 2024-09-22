# SeisComp EventXML Generator

This tool generates synthetic seismic event data in SeisComp XML format, useful for testing and development purposes in seismological applications.

## Table of Contents
- [SeisComp EventXML Generator](#seiscomp-eventxml-generator)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Output](#output)
  - [GUI Usage](#gui-usage)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Features

- Creates synthetic seismic events with realistic parameters
- Generates multiple origins, magnitudes, and focal mechanisms
- Produces SeisComp-compatible XML output
- Configurable via INI file for easy customization
- Fetches station data from FDSN web services
- Calculates various magnitude types (MLv, mb, Ms(BB), Mwp, Mw, Mww)
- Creates focal mechanisms with associated moment tensors
- Provides a Graphical User Interface (GUI) for easy configuration and execution

## Requirements

- Python 3.7+
- SeisComp3 Python environment
- Required Python packages:
  - numpy
  - configparser
  - requests
  - obspy
  - folium (for GUI map functionality)
  - PyQt5 (for GUI)

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:comoglu/seiscomp-eventxml-generator.git
   ```
2. Navigate to the project directory:
   ```
   cd seiscomp-eventxml-generator
   ```
3. Install the required Python packages:
   ```
   pip install numpy configparser requests obspy folium PyQt5
   ```
4. Ensure SeisComp3 is installed and its Python libraries are in your Python path.

## Configuration

The `config.ini` file allows you to customize various aspects of the synthetic event:

- Event parameters (location, time, type)
- Magnitude values for different types
- Focal mechanism parameters (strike, dip, rake)
- Inventory settings
- Noise levels
- Agency information
- Uncertainties
- Quality parameters
- FDSN web service URL
- Multiple origins settings

Refer to the provided `config.ini` for an example configuration.

## Usage

To generate a synthetic event, run:

```
seiscomp-python seiscomp-eventxml-generator-v2.py config.ini inventory.xml
```

This will create a SeisComP XML file containing the synthetic event based on the parameters in `config.ini`.

## Output

The generator produces a SeisComP XML file named `synthetic_event_seiscomp_[EVENT_ID].xml`, where `[EVENT_ID]` is a unique identifier for the event. This file contains:

- Event metadata
- Multiple origins with associated arrivals and magnitudes
- Focal mechanisms with moment tensors
- Station magnitudes
- Mww magnitudes linked to centroids

A summary of the generated event is printed to the console for quick reference.

## GUI Usage

To use the graphical interface:

1. Run the GUI script:
   ```
   python SC_GUI.py
   ```
2. Use the interface to modify event parameters, magnitudes, and focal mechanisms.
3. Click "Save Config" to update the `config.ini` file.
4. Click "Run Generator" to create the synthetic event.
5. The "Dispatch Event" button can be used to send the generated event to a SeisComP system (requires proper setup).

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Check that SeisComP Python libraries are in your Python path.
- Verify that the `config.ini` and `inventory.xml` files are in the correct locations.
- For GUI issues, ensure PyQt5 is properly installed.
- If you encounter any "module not found" errors, make sure all required packages are installed.

## Contributing

Contributions to improve the SeisComp EventXML Generator are welcome. Please feel free to submit issues or pull requests on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

comoglu AT gmail.com
