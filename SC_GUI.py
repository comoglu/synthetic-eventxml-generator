import sys
import os
import configparser
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGridLayout,
                             QTabWidget, QGroupBox, QMessageBox, QScrollArea)
from PyQt5.QtCore import QProcess, Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from obspy.imaging.beachball import beach
import folium
import io

class Bridge(QObject):
    locationChanged = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def updateLocation(self, lat, lng):
        print(f"Location updated: {lat}, {lng}")
        self.locationChanged.emit(lat, lng)

class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        self.setLayout(layout)

        coordinate = (0, 0)
        self.map = folium.Map(
            zoom_start=2,
            location=coordinate
        )

        self.marker = folium.Marker(
            coordinate,
            draggable=True
        )
        self.marker.add_to(self.map)

        self.webView = QWebEngineView()
        self.channel = QWebChannel()
        self.bridge = Bridge()
        self.channel.registerObject('pyObj', self.bridge)
        self.webView.page().setWebChannel(self.channel)

        self.bridge.locationChanged.connect(self.parent.update_location)

        layout.addWidget(self.webView)

        self.update_map(coordinate)

    def update_map(self, coordinate):
        self.map.location = coordinate
        self.marker.location = coordinate
        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())

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

class BeachballWidget(QWidget):
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
        self.ax.clear()
        b = beach([self.strike, self.dip, self.rake], width=200, linewidth=1, facecolor='r')
        self.ax.add_collection(b)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-105, 105)
        self.ax.set_ylim(-105, 105)
        self.ax.axis('off')
        self.canvas.draw()

    def mousePressEvent(self, event):
        self.update_focal_mechanism(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.update_focal_mechanism(event)

    def update_focal_mechanism(self, event):
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

class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr

class SeismicEventGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.config = CaseSensitiveConfigParser()
        self.load_config()
        self.output_file = ""


    def init_ui(self):
        self.setWindowTitle('Seismic Event Generator')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Event tab
        event_tab = QScrollArea()
        event_widget = QWidget()
        event_layout = QGridLayout()
        event_widget.setLayout(event_layout)
        event_tab.setWidget(event_widget)
        event_tab.setWidgetResizable(True)
        tabs.addTab(event_tab, "Event")

        self.event_edits = {}
        event_params = ['latitude', 'longitude', 'depth', 'time', 'type']
        for i, param in enumerate(event_params):
            event_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.event_edits[param] = QLineEdit()
            if param in ['latitude', 'longitude']:
                self.event_edits[param].textChanged.connect(self.update_map_from_input)
            event_layout.addWidget(self.event_edits[param], i, 1)

        self.map_widget = MapWidget(self)
        event_layout.addWidget(self.map_widget, 0, 2, len(event_params), 1)

        # Update Magnitudes tab
        magnitudes_tab = QScrollArea()
        magnitudes_widget = QWidget()
        magnitudes_layout = QGridLayout()
        magnitudes_widget.setLayout(magnitudes_layout)
        magnitudes_tab.setWidget(magnitudes_widget)
        magnitudes_tab.setWidgetResizable(True)
        tabs.addTab(magnitudes_tab, "Magnitudes")

        self.magnitude_edits = {}
        self.magnitude_types = []



        # Focal Mechanism tab
        focal_tab = QScrollArea()
        focal_widget = QWidget()
        focal_layout = QGridLayout()
        focal_widget.setLayout(focal_layout)
        focal_tab.setWidget(focal_widget)
        focal_tab.setWidgetResizable(True)
        tabs.addTab(focal_tab, "Focal Mechanism")

        self.focal_edits = {}
        focal_params = ['count']
        for i in range(1, 4):  # Assuming up to 3 focal mechanisms
            focal_params.extend([f'strike1_{i}', f'dip1_{i}', f'rake1_{i}', f'strike2_{i}', f'dip2_{i}', f'rake2_{i}'])
        for i, param in enumerate(focal_params):
            focal_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.focal_edits[param] = QLineEdit()
            if param in ['strike1_1', 'dip1_1', 'rake1_1']:
                self.focal_edits[param].textChanged.connect(self.update_beachball)
            focal_layout.addWidget(self.focal_edits[param], i, 1)

        self.beachball = BeachballWidget(self)
        focal_layout.addWidget(self.beachball, 0, 2, len(focal_params), 1)

        # Inventory tab
        inventory_tab = QScrollArea()
        inventory_widget = QWidget()
        inventory_layout = QGridLayout()
        inventory_widget.setLayout(inventory_layout)
        inventory_tab.setWidget(inventory_widget)
        inventory_tab.setWidgetResizable(True)
        tabs.addTab(inventory_tab, "Inventory")

        self.inventory_edits = {}
        inventory_params = ['path', 'min_distance', 'max_distance']
        for i, param in enumerate(inventory_params):
            inventory_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.inventory_edits[param] = QLineEdit()
            inventory_layout.addWidget(self.inventory_edits[param], i, 1)

        # Noise tab
        noise_tab = QScrollArea()
        noise_widget = QWidget()
        noise_layout = QGridLayout()
        noise_widget.setLayout(noise_layout)
        noise_tab.setWidget(noise_widget)
        noise_tab.setWidgetResizable(True)
        tabs.addTab(noise_tab, "Noise")

        self.noise_edits = {}
        noise_params = ['pick_time_std', 'station_magnitude_std']
        for i, param in enumerate(noise_params):
            noise_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.noise_edits[param] = QLineEdit()
            noise_layout.addWidget(self.noise_edits[param], i, 1)

        # Agency tab
        agency_tab = QScrollArea()
        agency_widget = QWidget()
        agency_layout = QGridLayout()
        agency_widget.setLayout(agency_layout)
        agency_tab.setWidget(agency_widget)
        agency_tab.setWidgetResizable(True)
        tabs.addTab(agency_tab, "Agency")

        self.agency_edits = {}
        agency_params = ['id', 'id_lowercase']
        for i, param in enumerate(agency_params):
            agency_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.agency_edits[param] = QLineEdit()
            agency_layout.addWidget(self.agency_edits[param], i, 1)

        # Uncertainties tab
        uncertainties_tab = QScrollArea()
        uncertainties_widget = QWidget()
        uncertainties_layout = QGridLayout()
        uncertainties_widget.setLayout(uncertainties_layout)
        uncertainties_tab.setWidget(uncertainties_widget)
        uncertainties_tab.setWidgetResizable(True)
        tabs.addTab(uncertainties_tab, "Uncertainties")

        self.uncertainty_edits = {}
        uncertainty_params = ['origin_latitude', 'origin_longitude', 'origin_depth']
        for i, param in enumerate(uncertainty_params):
            uncertainties_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.uncertainty_edits[param] = QLineEdit()
            uncertainties_layout.addWidget(self.uncertainty_edits[param], i, 1)

        # Phases tab
        phases_tab = QScrollArea()
        phases_widget = QWidget()
        phases_layout = QGridLayout()
        phases_widget.setLayout(phases_layout)
        phases_tab.setWidget(phases_widget)
        phases_tab.setWidgetResizable(True)
        tabs.addTab(phases_tab, "Phases")

        self.phases_edits = {}
        phases_params = ['s_wave_cutoff']
        for i, param in enumerate(phases_params):
            phases_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.phases_edits[param] = QLineEdit()
            phases_layout.addWidget(self.phases_edits[param], i, 1)

        # Add Mww field
        magnitudes_layout.addWidget(QLabel("Mww (comma-separated):"), 0, 0)
        self.magnitude_edits['Mww'] = QLineEdit()
        magnitudes_layout.addWidget(self.magnitude_edits['Mww'], 0, 1)

        # Update Multiple Origins tab
        multiple_origins_tab = QScrollArea()
        multiple_origins_widget = QWidget()
        multiple_origins_layout = QGridLayout()
        multiple_origins_widget.setLayout(multiple_origins_layout)
        multiple_origins_tab.setWidget(multiple_origins_widget)
        multiple_origins_tab.setWidgetResizable(True)
        tabs.addTab(multiple_origins_tab, "Multiple Origins")

        self.multiple_origins_edits = {}
        multiple_origins_params = ['number_of_origins', 'initial_station_count', 'station_increase_per_origin', 'creation_time_increment']
        for i, param in enumerate(multiple_origins_params):
            multiple_origins_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.multiple_origins_edits[param] = QLineEdit()
            multiple_origins_layout.addWidget(self.multiple_origins_edits[param], i, 1)


        # Quality Parameters tab
        quality_tab = QScrollArea()
        quality_widget = QWidget()
        quality_layout = QGridLayout()
        quality_widget.setLayout(quality_layout)
        quality_tab.setWidget(quality_widget)
        quality_tab.setWidgetResizable(True)
        tabs.addTab(quality_tab, "Quality Parameters")

        self.quality_edits = {}
        quality_params = ['standard_error', 'secondary_azimuthal_gap', 'ground_truth_level', 'maximum_distance', 'minimum_distance', 'median_distance']
        for i, param in enumerate(quality_params):
            quality_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.quality_edits[param] = QLineEdit()
            quality_layout.addWidget(self.quality_edits[param], i, 1)

        # FDSN tab
        fdsn_tab = QScrollArea()
        fdsn_widget = QWidget()
        fdsn_layout = QGridLayout()
        fdsn_widget.setLayout(fdsn_layout)
        fdsn_tab.setWidget(fdsn_widget)
        fdsn_tab.setWidgetResizable(True)
        tabs.addTab(fdsn_tab, "FDSN")

        self.fdsn_edits = {}
        fdsn_params = ['url']
        for i, param in enumerate(fdsn_params):
            fdsn_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.fdsn_edits[param] = QLineEdit()
            fdsn_layout.addWidget(self.fdsn_edits[param], i, 1)

        # Buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        save_config_button = QPushButton("Save Config")
        save_config_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_config_button)

        run_button = QPushButton("Run Generator")
        run_button.clicked.connect(self.run_generator)
        button_layout.addWidget(run_button)

        self.dispatch_button = QPushButton("Dispatch Event")
        self.dispatch_button.clicked.connect(self.dispatch_event)
        self.dispatch_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.dispatch_button)

        # Output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

    def load_config(self):
        if os.path.exists('config.ini'):
            self.config.read('config.ini')

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
                for param, edit in edits.items():
                    edit.setText(self.config.get(section, param, fallback=''))
            
            # Handle Magnitudes separately to preserve case
            if 'Magnitudes' in self.config:
                self.magnitude_types = list(self.config['Magnitudes'].keys())
                magnitudes_layout = self.centralWidget().findChild(QTabWidget).widget(1).widget().layout()
                for i, mag_type in enumerate(self.magnitude_types):
                    if mag_type != 'Mww':  # We've already added Mww field separately
                        magnitudes_layout.addWidget(QLabel(f"{mag_type}:"), i+1, 0)  # Start from row 1
                        self.magnitude_edits[mag_type] = QLineEdit()
                        self.magnitude_edits[mag_type].setText(self.config.get('Magnitudes', mag_type, fallback=''))
                        magnitudes_layout.addWidget(self.magnitude_edits[mag_type], i+1, 1)  # Start from row 1
                
                # Set Mww values
                self.magnitude_edits['Mww'].setText(self.config.get('Magnitudes', 'Mww', fallback=''))
            
            # Handle FocalMechanism separately
            if 'FocalMechanism' in self.config:
                for param, edit in self.focal_edits.items():
                    edit.setText(self.config.get('FocalMechanism', param, fallback=''))
        else:
            QMessageBox.warning(self, "Config Error", "config.ini file not found. Using default values.")


    def save_config(self):
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

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        self.output_text.append("Configuration saved to config.ini")



    def update_beachball(self):
        try:
            strike = float(self.focal_edits['strike1_1'].text())
            dip = float(self.focal_edits['dip1_1'].text())
            rake = float(self.focal_edits['rake1_1'].text())
            self.beachball.strike, self.beachball.dip, self.beachball.rake = strike, dip, rake
            self.beachball.draw_beachball()
        except ValueError:
            pass

    def update_focal_mechanism_values(self, strike, dip, rake):
        self.focal_edits['strike1_1'].setText(f"{strike:.2f}")
        self.focal_edits['dip1_1'].setText(f"{dip:.2f}")
        self.focal_edits['rake1_1'].setText(f"{rake:.2f}")

    def run_generator(self):
        self.save_config()
        self.output_text.clear()
        self.output_text.append("Running Seismic Event Generator...")
        
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self.handle_stdout)
        process.readyReadStandardError.connect(self.handle_stderr)
        process.finished.connect(self.process_finished)
        
        # Use the full path to the Python interpreter and script
        python_path = sys.executable
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seiscomp-eventxml-generator.py")
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        inventory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.xml")
        
        process.start(python_path, [script_path, config_path, inventory_path])

    def handle_stdout(self):
        process = self.sender()
        stdout = process.readAllStandardOutput()
        output = bytes(stdout).decode("utf8")
        self.output_text.append(output)
        
        # Check if the output contains the name of the generated file
        for line in output.split('\n'):
            if line.startswith("Event has been written to "):
                self.output_file = line.split("to ")[-1].strip()
                break

    def handle_stderr(self):
        process = self.sender()
        stderr = process.readAllStandardError()
        output = bytes(stderr).decode("utf8")
        self.output_text.append(output)

    def process_finished(self):
        self.output_text.append("Seismic Event Generator finished execution.")
        if self.output_file:
            self.dispatch_button.setEnabled(True)
        else:
            self.output_text.append("Warning: No output file was detected.")

    def dispatch_event(self):
        if not self.output_file:
            QMessageBox.warning(self, "Error", "No output file available to dispatch.")
            return
        
        self.output_text.append(f"Dispatching event from file: {self.output_file}")
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self.handle_stdout)
        process.readyReadStandardError.connect(self.handle_stderr)
        process.finished.connect(self.dispatch_finished)
        process.start("scdispatch", ["-i", self.output_file])

    def dispatch_finished(self):
        self.output_text.append("Event dispatch process completed.")
        self.dispatch_button.setEnabled(False)
        self.output_file = ""

    def update_location(self, lat, lng):
        self.event_edits['latitude'].setText(f"{lat:.4f}")
        self.event_edits['longitude'].setText(f"{lng:.4f}")
        print(f"Location updated in GUI: {lat:.4f}, {lng:.4f}")

    def update_map_from_input(self):
        try:
            lat = float(self.event_edits['latitude'].text())
            lon = float(self.event_edits['longitude'].text())
            self.map_widget.update_map((lat, lon))
        except ValueError:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SeismicEventGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())
