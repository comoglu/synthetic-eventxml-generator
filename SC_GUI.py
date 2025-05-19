#!/usr/bin/python3

import sys
import os
import configparser
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGridLayout,
                             QTabWidget, QGroupBox, QMessageBox, QScrollArea, QSplitter,
                             QStatusBar, QFrame)
from PyQt5.QtCore import QProcess, Qt, QUrl
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from obspy.imaging.beachball import beach
import folium
import io

# Set up logging
def setup_logging():
    logger = logging.getLogger('seismic_gui')
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler('seismic_gui.log')
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

logger = setup_logging()

# Constants for styling
STYLE_SHEET = """
QMainWindow {
    background-color: #f0f0f0;
}
QPushButton {
    background-color: #2980b9;
    color: white;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3498db;
}
QPushButton:disabled {
    background-color: #95a5a6;
}
QTabWidget::pane {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #ecf0f1;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #3498db;
    color: white;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    margin-top: 1em;
    padding-top: 1em;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QTextEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: #ffffff;
}
QLineEdit {
    padding: 5px;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
}
"""

class Bridge(QObject):
    """Bridge class for communication between Python and JavaScript in the map widget."""
    locationChanged = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def updateLocation(self, lat, lng):
        """Update location when map marker is moved or clicked."""
        logger.debug(f"Location updated: {lat}, {lng}")
        self.locationChanged.emit(lat, lng)


class MapWidget(QWidget):
    """Interactive map widget using Folium and QWebEngineView."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Initialize map with default coordinates
        coordinate = (0, 0)
        self.map = folium.Map(
            zoom_start=2,
            location=coordinate
        )

        # Add a draggable marker
        self.marker = folium.Marker(
            coordinate,
            draggable=True
        )
        self.marker.add_to(self.map)

        # Set up web view and JavaScript communication
        self.webView = QWebEngineView()
        self.channel = QWebChannel()
        self.bridge = Bridge()
        self.channel.registerObject('pyObj', self.bridge)
        self.webView.page().setWebChannel(self.channel)

        # Connect bridge signal to parent's update method
        self.bridge.locationChanged.connect(self.parent.update_location)

        layout.addWidget(self.webView)
        self.update_map(coordinate)

    def update_map(self, coordinate):
        """Update the map with new coordinates."""
        try:
            self.map.location = coordinate
            self.marker.location = coordinate
            data = io.BytesIO()
            self.map.save(data, close_file=False)
            self.webView.setHtml(data.getvalue().decode())

            # JavaScript to initialize map and handle events
            self.webView.page().runJavaScript('''
                function initMap() {
                    var map = document.getElementsByTagName('div')[0];
                    if (!map) {
                        console.error('Map div not found');
                        return;
                    }
                    map.style.height = '100%';
                    map.style.width = '100%';
                    if (map.leaflet_map) {
                        map.leaflet_map.invalidateSize();
                        map.leaflet_map.on('click', function(e) {
                            if (window.pyObj) {
                                window.pyObj.updateLocation(e.latlng.lat, e.latlng.lng);
                            }
                        });
                        var marker = map.leaflet_map.markers[0];
                        if (marker) {
                            marker.on('dragend', function(e) {
                                if (window.pyObj) {
                                    window.pyObj.updateLocation(e.target._latlng.lat, e.target._latlng.lng);
                                }
                            });
                        } else {
                            console.error('Marker not found');
                        }
                    } else {
                        console.error('Leaflet map not initialized');
                    }
                }

                setTimeout(initMap, 500);
            ''')
        except Exception as e:
            logger.error(f"Error updating map: {e}")
            # Show error to user
            QMessageBox.warning(self, "Map Error", f"Error updating map: {e}")


class BeachballWidget(QWidget):
    """Interactive focal mechanism beachball widget."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111)
        self.strike, self.dip, self.rake = 0, 90, 0
        self.draw_beachball()

    def draw_beachball(self):
        """Draw focal mechanism beachball."""
        try:
            self.ax.clear()
            b = beach([self.strike, self.dip, self.rake], width=200, linewidth=1, facecolor='r')
            self.ax.add_collection(b)
            self.ax.set_aspect("equal")
            self.ax.set_xlim(-105, 105)
            self.ax.set_ylim(-105, 105)
            self.ax.axis('off')
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error drawing beachball: {e}")

    def mousePressEvent(self, event):
        """Handle mouse press events on the beachball."""
        self.update_focal_mechanism(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events on the beachball."""
        if event.buttons() & Qt.LeftButton:
            self.update_focal_mechanism(event)

    def update_focal_mechanism(self, event):
        """Update focal mechanism parameters based on mouse position."""
        try:
            if self.ax.contains(event)[0]:
                x, y = self.ax.transData.inverted().transform([event.x(), event.y()])
                r = np.sqrt(x**2 + y**2)
                if r <= 100:
                    azimuth = np.degrees(np.arctan2(x, y)) % 360
                    plunge = 90 - r * 90 / 100
                    
                    if event.modifiers() & Qt.ShiftModifier:
                        self.dip = plunge
                        self.rake = azimuth
                    else:
                        self.strike = azimuth
                        self.dip = plunge

                    self.draw_beachball()
                    self.parent().update_focal_mechanism_values(self.strike, self.dip, self.rake)
        except Exception as e:
            logger.error(f"Error updating focal mechanism: {e}")


class CaseSensitiveConfigParser(configparser.ConfigParser):
    """ConfigParser that preserves case of keys."""
    def optionxform(self, optionstr):
        return optionstr


class ScrollableTabWidget(QScrollArea):
    """A scrollable container for tab widgets to handle many fields."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.widget = QWidget()
        self.layout = QGridLayout(self.widget)
        self.setWidget(self.widget)
        self.fields = {}

    def add_field(self, name, row):
        """Add a field with label to the layout."""
        self.layout.addWidget(QLabel(f"{name}:"), row, 0)
        self.fields[name] = QLineEdit()
        self.layout.addWidget(self.fields[name], row, 1)
        return self.fields[name]


class SeismicEventGeneratorGUI(QMainWindow):
    """Main application window for the Seismic Event Generator GUI."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Seismic Event Generator')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLE_SHEET)
        
        # Initialize config
        self.config = CaseSensitiveConfigParser()
        
        # Initialize UI components
        self.init_ui()
        
        # Load configuration
        self.load_config()
        
        # Initialize output file
        self.output_file = ""
        
        # Set up status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready", 5000)

    def init_ui(self):
        """Initialize the user interface."""
        # Set up main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Initialize all tab widgets with scrolling capability
        self.setup_event_tab()
        self.setup_magnitudes_tab()
        self.setup_focal_mechanism_tab()
        self.setup_inventory_tab()
        self.setup_noise_tab()
        self.setup_agency_tab()
        self.setup_uncertainties_tab()
        self.setup_phases_tab()
        self.setup_multiple_origins_tab()
        self.setup_quality_parameters_tab()
        self.setup_fdsn_tab()

        # Button layout
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        # Save config button
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_config_button)

        # Run generator button
        self.run_button = QPushButton("Run Generator")
        self.run_button.clicked.connect(self.run_generator)
        button_layout.addWidget(self.run_button)

        # Dispatch button
        self.dispatch_button = QPushButton("Dispatch Event")
        self.dispatch_button.clicked.connect(self.dispatch_event)
        self.dispatch_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.dispatch_button)

        # Output area with improved styling
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(200)
        output_layout.addWidget(self.output_text)
        
        main_layout.addWidget(output_group)

    def setup_event_tab(self):
        """Set up the Event tab."""
        event_tab = ScrollableTabWidget()
        self.event_edits = {}
        
        event_params = ['latitude', 'longitude', 'depth', 'time', 'type']
        for i, param in enumerate(event_params):
            field = event_tab.add_field(param, i)
            self.event_edits[param] = field
            if param in ['latitude', 'longitude']:
                field.textChanged.connect(self.update_map_from_input)
        
        # Add map widget
        self.map_widget = MapWidget(self)
        event_tab.layout.addWidget(self.map_widget, 0, 2, len(event_params), 1)
        
        # Add help text
        help_label = QLabel("Click or drag the marker to set the event location")
        help_label.setWordWrap(True)
        event_tab.layout.addWidget(help_label, len(event_params), 0, 1, 2)
        
        self.tabs.addTab(event_tab, "Event")

    def setup_magnitudes_tab(self):
        """Set up the Magnitudes tab."""
        magnitudes_tab = ScrollableTabWidget()
        self.magnitude_edits = {}
        self.magnitude_types = []
        
        # Add Mww field
        field = magnitudes_tab.add_field("Mww (comma-separated)", 0)
        self.magnitude_edits['Mww'] = field
        
        # Empty space for other magnitude types to be loaded from config
        self.magnitudes_tab = magnitudes_tab
        self.tabs.addTab(magnitudes_tab, "Magnitudes")

    def setup_focal_mechanism_tab(self):
        """Set up the Focal Mechanism tab."""
        focal_tab = ScrollableTabWidget()
        self.focal_edits = {}
        
        focal_params = ['count']
        for i in range(1, 4):  # Assuming up to 3 focal mechanisms
            focal_params.extend([f'strike1_{i}', f'dip1_{i}', f'rake1_{i}', 
                               f'strike2_{i}', f'dip2_{i}', f'rake2_{i}'])
        
        for i, param in enumerate(focal_params):
            field = focal_tab.add_field(param, i)
            self.focal_edits[param] = field
            if param in ['strike1_1', 'dip1_1', 'rake1_1']:
                field.textChanged.connect(self.update_beachball)
        
        # Add beachball widget
        self.beachball = BeachballWidget(self)
        focal_tab.layout.addWidget(self.beachball, 0, 2, len(focal_params), 1)
        
        # Add help text
        help_label = QLabel("Click and drag on the beachball to adjust strike and dip.\nHold Shift to adjust dip and rake.")
        help_label.setWordWrap(True)
        focal_tab.layout.addWidget(help_label, len(focal_params), 0, 1, 2)
        
        self.tabs.addTab(focal_tab, "Focal Mechanism")

    def setup_inventory_tab(self):
        """Set up the Inventory tab."""
        inventory_tab = ScrollableTabWidget()
        self.inventory_edits = {}
        
        inventory_params = ['path', 'min_distance', 'max_distance']
        for i, param in enumerate(inventory_params):
            self.inventory_edits[param] = inventory_tab.add_field(param, i)
        
        # Add file browser button
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_inventory)
        inventory_tab.layout.addWidget(browse_button, 0, 2)
        
        self.tabs.addTab(inventory_tab, "Inventory")

    def setup_noise_tab(self):
        """Set up the Noise tab."""
        noise_tab = ScrollableTabWidget()
        self.noise_edits = {}
        
        noise_params = ['pick_time_std', 'station_magnitude_std']
        for i, param in enumerate(noise_params):
            self.noise_edits[param] = noise_tab.add_field(param, i)
        
        self.tabs.addTab(noise_tab, "Noise")

    def setup_agency_tab(self):
        """Set up the Agency tab."""
        agency_tab = ScrollableTabWidget()
        self.agency_edits = {}
        
        agency_params = ['id', 'id_lowercase']
        for i, param in enumerate(agency_params):
            self.agency_edits[param] = agency_tab.add_field(param, i)
        
        self.tabs.addTab(agency_tab, "Agency")

    def setup_uncertainties_tab(self):
        """Set up the Uncertainties tab."""
        uncertainties_tab = ScrollableTabWidget()
        self.uncertainty_edits = {}
        
        uncertainty_params = ['origin_latitude', 'origin_longitude', 'origin_depth']
        for i, param in enumerate(uncertainty_params):
            self.uncertainty_edits[param] = uncertainties_tab.add_field(param, i)
        
        self.tabs.addTab(uncertainties_tab, "Uncertainties")

    def setup_phases_tab(self):
        """Set up the Phases tab."""
        phases_tab = ScrollableTabWidget()
        self.phases_edits = {}
        
        phases_params = ['s_wave_cutoff']
        for i, param in enumerate(phases_params):
            self.phases_edits[param] = phases_tab.add_field(param, i)
        
        self.tabs.addTab(phases_tab, "Phases")

    def setup_multiple_origins_tab(self):
        """Set up the Multiple Origins tab."""
        multiple_origins_tab = ScrollableTabWidget()
        self.multiple_origins_edits = {}
        
        multiple_origins_params = [
            'number_of_origins', 
            'initial_station_count', 
            'station_increase_per_origin', 
            'creation_time_increment'
        ]
        
        for i, param in enumerate(multiple_origins_params):
            self.multiple_origins_edits[param] = multiple_origins_tab.add_field(param, i)
        
        self.tabs.addTab(multiple_origins_tab, "Multiple Origins")

    def setup_quality_parameters_tab(self):
        """Set up the Quality Parameters tab."""
        quality_tab = ScrollableTabWidget()
        self.quality_edits = {}
        
        quality_params = [
            'standard_error', 
            'secondary_azimuthal_gap', 
            'ground_truth_level', 
            'maximum_distance', 
            'minimum_distance', 
            'median_distance'
        ]
        
        for i, param in enumerate(quality_params):
            self.quality_edits[param] = quality_tab.add_field(param, i)
        
        self.tabs.addTab(quality_tab, "Quality Parameters")

    def setup_fdsn_tab(self):
        """Set up the FDSN tab."""
        fdsn_tab = ScrollableTabWidget()
        self.fdsn_edits = {}
        
        fdsn_params = ['url']
        for i, param in enumerate(fdsn_params):
            self.fdsn_edits[param] = fdsn_tab.add_field(param, i)
        
        self.tabs.addTab(fdsn_tab, "FDSN")

    def browse_inventory(self):
        """Open file browser to select inventory file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Inventory File", "", "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.inventory_edits['path'].setText(file_path)
            self.log(f"Selected inventory file: {file_path}")

    def log(self, message, error=False):
        """Add message to log with optional styling."""
        if error:
            self.output_text.append(f"<span style='color:red'>{message}</span>")
        else:
            self.output_text.append(message)
        
        # Auto-scroll to bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)
        
        # Also log to file
        if error:
            logger.error(message)
        else:
            logger.info(message)

    def load_config(self):
        """Load configuration and update UI fields."""
        try:
            if os.path.exists('config.ini'):
                self.config.read('config.ini')

                # Handle sections with straightforward mapping to edit widgets
                for section, edits in [
                    ('Event', self.event_edits),
                    ('Inventory', self.inventory_edits),
                    ('Noise', self.noise_edits),
                    ('Agency', self.agency_edits),
                    ('Uncertainties', self.uncertainty_edits),
                    ('Phases', self.phases_edits),
                    ('MultipleOrigins', self.multiple_origins_edits),
                    ('QualityParameters', self.quality_edits),
                    ('FDSN', self.fdsn_edits)
                ]:
                    if section in self.config:
                        for param, edit in edits.items():
                            edit.setText(self.config.get(section, param, fallback=''))
                
                # Handle Magnitudes separately to preserve case
                if 'Magnitudes' in self.config:
                    self.magnitude_types = list(self.config['Magnitudes'].keys())
                    
                    # Set Mww value which has a dedicated field
                    self.magnitude_edits['Mww'].setText(self.config.get('Magnitudes', 'Mww', fallback=''))
                    
                    # Add other magnitude types dynamically
                    for i, mag_type in enumerate(self.magnitude_types):
                        if mag_type != 'Mww':  # Skip Mww as it's already added
                            row = i + 1  # Start from row 1
                            self.magnitudes_tab.layout.addWidget(QLabel(f"{mag_type}:"), row, 0)
                            self.magnitude_edits[mag_type] = QLineEdit()
                            self.magnitude_edits[mag_type].setText(self.config.get('Magnitudes', mag_type, fallback=''))
                            self.magnitudes_tab.layout.addWidget(self.magnitude_edits[mag_type], row, 1)
                
                # Handle FocalMechanism separately
                if 'FocalMechanism' in self.config:
                    for param, edit in self.focal_edits.items():
                        edit.setText(self.config.get('FocalMechanism', param, fallback=''))
                    
                    # Update beachball if strike/dip/rake are set
                    self.update_beachball()
                    
                # Update map if latitude/longitude are set
                self.update_map_from_input()
                
                self.log("Configuration loaded")
            else:
                self.log("No config.ini file found. Using default values.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.log(f"Error loading configuration: {e}", error=True)
            QMessageBox.warning(self, "Config Error", 
                              f"Error loading configuration: {e}\nUsing default values.")

    def save_config(self):
        """Save configuration from UI fields to file."""
        try:
            # Handle sections with straightforward mapping from edit widgets
            for section, edits in [
                ('Event', self.event_edits),
                ('Inventory', self.inventory_edits),
                ('Noise', self.noise_edits),
                ('Agency', self.agency_edits),
                ('Uncertainties', self.uncertainty_edits),
                ('Phases', self.phases_edits),
                ('MultipleOrigins', self.multiple_origins_edits),
                ('QualityParameters', self.quality_edits),
                ('FDSN', self.fdsn_edits)
            ]:
                if section not in self.config:
                    self.config[section] = {}
                for param, edit in edits.items():
                    self.config[section][param] = edit.text()
            
            # Handle Magnitudes separately to preserve case
            if 'Magnitudes' not in self.config:
                self.config['Magnitudes'] = {}
            for mag_type, edit in self.magnitude_edits.items():
                self.config['Magnitudes'][mag_type] = edit.text()
            
            # Handle FocalMechanism separately
            if 'FocalMechanism' not in self.config:
                self.config['FocalMechanism'] = {}
            for param, edit in self.focal_edits.items():
                self.config['FocalMechanism'][param] = edit.text()

            # Save the configuration
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
            
            self.log("Configuration saved to config.ini")
            self.statusBar.showMessage("Configuration saved", 3000)
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            self.log(f"Error saving configuration: {e}", error=True)
            QMessageBox.critical(self, "Config Error", 
                               f"Could not save configuration: {e}")

    def update_beachball(self):
        """Update the beachball widget from UI fields."""
        try:
            strike_field = self.focal_edits['strike1_1']
            dip_field = self.focal_edits['dip1_1']
            rake_field = self.focal_edits['rake1_1']
            
            if not all([strike_field.text(), dip_field.text(), rake_field.text()]):
                return  # Skip if any field is empty
                
            strike = float(strike_field.text())
            dip = float(dip_field.text())
            rake = float(rake_field.text())
            
            self.beachball.strike, self.beachball.dip, self.beachball.rake = strike, dip, rake
            self.beachball.draw_beachball()
            
        except ValueError:
            # Silently ignore conversion errors during typing
            pass
        except Exception as e:
            logger.error(f"Error updating beachball: {e}")

    def update_focal_mechanism_values(self, strike, dip, rake):
        """Update focal mechanism fields from the beachball widget."""
        try:
            self.focal_edits['strike1_1'].setText(f"{strike:.2f}")
            self.focal_edits['dip1_1'].setText(f"{dip:.2f}")
            self.focal_edits['rake1_1'].setText(f"{rake:.2f}")
            self.statusBar.showMessage(f"Focal mechanism updated: Strike={strike:.2f}, Dip={dip:.2f}, Rake={rake:.2f}", 3000)
        except Exception as e:
            logger.error(f"Error updating focal mechanism values: {e}")

    def run_generator(self):
        """Run the seismic event generator script."""
        # Save configuration before running
        self.save_config()
        
        # Clear output
        self.output_text.clear()
        self.log("Running Seismic Event Generator...")
        self.statusBar.showMessage("Generator running...", 0)  # 0 means no timeout
        
        # Disable buttons while running
        self.run_button.setEnabled(False)
        self.dispatch_button.setEnabled(False)
        
        try:
            # Set up QProcess
            process = QProcess(self)
            process.readyReadStandardOutput.connect(self.handle_stdout)
            process.readyReadStandardError.connect(self.handle_stderr)
            process.finished.connect(self.process_finished)
            
            # Use the full path to the Python interpreter and script
            python_path = sys.executable
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, "seiscomp-eventxml-generator.py")
            config_path = os.path.join(script_dir, "config.ini")
            
            # Ensure the inventory path is set correctly
            inventory_path = self.inventory_edits['path'].text()
            if not os.path.isabs(inventory_path):
                inventory_path = os.path.join(script_dir, inventory_path)
            
            # Start the process
            process.start(python_path, [script_path, config_path, inventory_path])
            
        except Exception as e:
            logger.error(f"Error starting generator: {e}")
            self.log(f"Error starting generator: {e}", error=True)
            self.run_button.setEnabled(True)
            self.statusBar.showMessage("Generator failed to start", 5000)

    def handle_stdout(self):
        """Handle standard output from the generator process."""
        process = self.sender()
        stdout = process.readAllStandardOutput()
        output = bytes(stdout).decode("utf8")
        self.log(output.strip())
        
        # Check if the output contains the name of the generated file
        for line in output.split('\n'):
            # Look for variations of output file notification
            if "Event has been written to " in line:
                self.output_file = line.split("to ")[-1].strip()
                logger.info(f"Output file detected: {self.output_file}")
                break
            elif "written to" in line and ".xml" in line:
                # More flexible detection for output file paths
                parts = line.split()
                for part in parts:
                    if part.endswith(".xml"):
                        self.output_file = part.strip()
                        logger.info(f"Output file detected: {self.output_file}")
                        break
                if self.output_file:
                    break

    def handle_stderr(self):
        """Handle standard error from the generator process."""
        process = self.sender()
        stderr = process.readAllStandardError()
        output = bytes(stderr).decode("utf8")
        self.log(output.strip(), error=True)

    def process_finished(self, exit_code, exit_status):
        """Handle process completion."""
        self.run_button.setEnabled(True)
        
        if exit_code == 0:
            self.log("Seismic Event Generator finished successfully.")
            
            # If output file wasn't detected in stdout, try to find it in the directory
            if not self.output_file:
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    xml_files = [f for f in os.listdir(script_dir) if f.startswith("synthetic_event_seiscomp_") and f.endswith(".xml")]
                    if xml_files:
                        # Get the most recently created file
                        latest_file = max(xml_files, key=lambda f: os.path.getmtime(os.path.join(script_dir, f)))
                        self.output_file = os.path.join(script_dir, latest_file)
                        self.log(f"Found output file by directory scan: {self.output_file}")
                except Exception as e:
                    logger.error(f"Failed to scan for output files: {e}")
            
            if self.output_file:
                self.dispatch_button.setEnabled(True)
                self.log(f"Output file: {self.output_file}")
                self.statusBar.showMessage("Generator completed successfully", 5000)
            else:
                self.log("Warning: No output file was detected.", error=True)
                self.statusBar.showMessage("Generator completed, but no output file found", 5000)
        else:
            self.log(f"Generator process exited with code {exit_code}.", error=True)
            self.statusBar.showMessage("Generator failed", 5000)

    def dispatch_event(self):
        """Dispatch the generated event to SeisComp."""
        if not self.output_file:
            QMessageBox.warning(self, "Error", "No output file available to dispatch.")
            return
        
        self.log(f"Dispatching event from file: {self.output_file}")
        self.statusBar.showMessage("Dispatching event...", 0)
        self.dispatch_button.setEnabled(False)
        
        try:
            process = QProcess(self)
            process.readyReadStandardOutput.connect(self.handle_stdout)
            process.readyReadStandardError.connect(self.handle_stderr)
            process.finished.connect(self.dispatch_finished)
            process.start("scdispatch", ["-i", self.output_file])
        except Exception as e:
            logger.error(f"Error starting dispatch: {e}")
            self.log(f"Error starting dispatch: {e}", error=True)
            self.dispatch_button.setEnabled(True)
            self.statusBar.showMessage("Dispatch failed", 5000)

    def dispatch_finished(self, exit_code, exit_status):
        """Handle event dispatch completion."""
        if exit_code == 0:
            self.log("Event dispatch completed successfully.")
            self.statusBar.showMessage("Event dispatched successfully", 5000)
        else:
            self.log(f"Event dispatch failed with exit code {exit_code}.", error=True)
            self.statusBar.showMessage("Event dispatch failed", 5000)
        
        self.dispatch_button.setEnabled(False)
        self.output_file = ""

    def update_location(self, lat, lng):
        """Update location fields when map marker is moved."""
        try:
            self.event_edits['latitude'].setText(f"{lat:.4f}")
            self.event_edits['longitude'].setText(f"{lng:.4f}")
            logger.debug(f"Location updated in GUI: {lat:.4f}, {lng:.4f}")
            self.statusBar.showMessage(f"Location updated: {lat:.4f}, {lng:.4f}", 3000)
        except Exception as e:
            logger.error(f"Error updating location: {e}")

    def update_map_from_input(self):
        """Update map marker when location fields are changed."""
        try:
            lat_text = self.event_edits['latitude'].text()
            lon_text = self.event_edits['longitude'].text()
            
            if not lat_text or not lon_text:
                return
                
            lat = float(lat_text)
            lon = float(lon_text)
            
            # Basic validation
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                self.map_widget.update_map((lat, lon))
        except ValueError:
            # Silently ignore conversion errors during typing
            pass
        except Exception as e:
            logger.error(f"Error updating map: {e}")

    def closeEvent(self, event):
        """Handle window close event."""
        # Ask user if they want to save before closing
        if self.isWindowModified():
            reply = QMessageBox.question(self, 'Save Configuration',
                                      'Do you want to save your configuration before exiting?',
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                      QMessageBox.Save)
            
            if reply == QMessageBox.Save:
                self.save_config()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        gui = SeismicEventGeneratorGUI()
        gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        if QApplication.instance():
            QMessageBox.critical(None, "Fatal Error", 
                             f"A fatal error occurred: {e}\n\nCheck the log file for details.")
