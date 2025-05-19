#!/usr/bin/env seiscomp-python

import sys
import random
import numpy as np
from seiscomp import core, datamodel, io, math as sc_math
import configparser
import datetime
import string
import requests
import csv
from io import StringIO
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, degrees2kilometers
import xml.etree.ElementTree as ET
import argparse
import logging
import os
from typing import List, Dict, Tuple, Any, Optional, Union

# Set up logging
def setup_logging():
    logger = logging.getLogger('seismic_generator')
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler('seismic_generator.log')
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

logger = setup_logging()

class ConfigManager:
    def __init__(self, config_file='config.ini'):
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self):
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve case
        if os.path.exists(self.config_file):
            config.read(self.config_file)
            logger.info(f"Loaded configuration from {self.config_file}")
        else:
            logger.warning(f"Configuration file {self.config_file} not found, using defaults")
            self.create_default_config(config)
        return config
    
    def create_default_config(self, config):
        # Set sensible defaults
        config['Event'] = {
            'latitude': '0.0',
            'longitude': '0.0',
            'depth': '10.0',
            'time': 'now',
            'type': 'EARTHQUAKE'
        }
        config['Inventory'] = {
            'path': 'inventory.xml',
            'min_distance': '0.0',
            'max_distance': '180.0'
        }
        config['Noise'] = {
            'pick_time_std': '0.1',
            'station_magnitude_std': '0.2'
        }
        config['Agency'] = {
            'id': 'TEST',
            'id_lowercase': 'test'
        }
        config['Uncertainties'] = {
            'origin_latitude': '0.1',
            'origin_longitude': '0.1',
            'origin_depth': '5.0'
        }
        config['Phases'] = {
            's_wave_cutoff': '105.0'
        }
        config['Magnitudes'] = {
            'Mww': '8.7, 9.1, 9.3',
            'ML': '6.5',
            'mb': '7.0',
            'mB': '7.5'
        }
        config['MultipleOrigins'] = {
            'number_of_origins': '3',
            'initial_station_count': '10',
            'station_increase_per_origin': '5',
            'creation_time_increment': '60.0'
        }
        config['QualityParameters'] = {
            'standard_error': '0.5',
            'secondary_azimuthal_gap': '60.0',
            'ground_truth_level': 'GT5',
            'maximum_distance': '90.0',
            'minimum_distance': '0.0',
            'median_distance': '45.0'
        }
        config['FocalMechanism'] = {
            'count': '1',
            'strike1_1': '300.0',
            'dip1_1': '15.0',
            'rake1_1': '90.0',
            'strike2_1': '120.0',
            'dip2_1': '75.0',
            'rake2_1': '90.0'
        }
        config['FDSN'] = {
            'url': 'http://localhost:8081/fdsnws/station/1/query'
        }
        return config
        
    def save_config(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)
            logger.info(f"Saved configuration to {self.config_file}")
        
    def get_config(self):
        return self.config

def create_resource_id(agency_id: str, id_string: str) -> str:
    """
    Create a resource ID string in SeisComp format.
    
    Args:
        agency_id: The agency identifier
        id_string: The resource identifier string
        
    Returns:
        Formatted resource ID
    """
    return f"smi:{agency_id.lower()}/{id_string}"

def parse_config_xml(config_file: str) -> List[Dict[str, str]]:
    """
    Parse the SeisComp configuration XML file to extract station information.
    
    Args:
        config_file: Path to the SeisComp configuration XML file
        
    Returns:
        List of dictionaries containing network and station codes
    """
    stations = []
    try:
        tree = ET.parse(config_file)
        root = tree.getroot()
        
        namespace = {'sc': 'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12'}
        
        for parameterSet in root.findall('.//sc:parameterSet', namespace):
            publicID = parameterSet.get('publicID', '')
            if publicID.startswith('ParameterSet/trunk/Station/'):
                parts = publicID.split('/')
                if len(parts) >= 5:
                    network = parts[3]
                    station = parts[4]
                    stations.append({
                        "network": network,
                        "station": station
                    })
        
        logger.info(f"Parsed {len(stations)} stations from configuration XML")
        return stations
    except Exception as e:
        logger.error(f"Error parsing configuration XML: {e}")
        return []

def filter_stations_by_distance(stations: List[Dict[str, Any]], max_distance: float) -> List[Dict[str, Any]]:
    """
    Filter stations based on distance from event.
    
    Args:
        stations: List of station dictionaries with distance information
        max_distance: Maximum distance in degrees
        
    Returns:
        Filtered list of stations within the maximum distance
    """
    filtered = [station for station in stations if station['distance'] <= max_distance]
    logger.debug(f"Filtered {len(filtered)} stations within {max_distance} degrees from {len(stations)} total stations")
    return filtered

def load_stations_from_inventory(inventory_path: str, event_latitude: float, event_longitude: float, 
                               max_distance: float) -> List[Dict[str, Any]]:
    """
    Load station information from the inventory XML file.
    
    Args:
        inventory_path: Path to the inventory XML file
        event_latitude: Event latitude in degrees
        event_longitude: Event longitude in degrees
        max_distance: Maximum distance in degrees
        
    Returns:
        List of station dictionaries with coordinates and calculated distances
    """
    stations = []
    try:
        tree = ET.parse(inventory_path)
        root = tree.getroot()
        namespace = {'sc': 'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12'}
        
        # First try to find networks and stations in SeisComp XML format
        for network in root.findall('.//sc:network', namespace):
            net_code = network.get('code')
            for station in network.findall('.//sc:station', namespace):
                sta_code = station.get('code')
                lat_elem = station.find('.//sc:latitude', namespace)
                lon_elem = station.find('.//sc:longitude', namespace)
                elev_elem = station.find('.//sc:elevation', namespace)
                
                if lat_elem is not None and lon_elem is not None:
                    lat = float(lat_elem.text)
                    lon = float(lon_elem.text)
                    elev = float(elev_elem.text) if elev_elem is not None else 0.0
                    
                    distance = locations2degrees(event_latitude, event_longitude, lat, lon)
                    if distance <= max_distance:
                        azimuth = sc_math.delazi(event_latitude, event_longitude, lat, lon)[1]
                        
                        stations.append({
                            "code": sta_code,
                            "network": net_code,
                            "latitude": lat,
                            "longitude": lon,
                            "elevation": elev,
                            "distance": distance,
                            "azimuth": azimuth
                        })
        
        # If no stations found with SC schema, try standard QuakeML/FDSN schema
        if not stations:
            for network in root.findall('.//network'):
                net_code = network.get('code')
                for station in network.findall('.//station'):
                    sta_code = station.get('code')
                    lat_elem = station.find('./latitude')
                    lon_elem = station.find('./longitude')
                    elev_elem = station.find('./elevation')
                    
                    if lat_elem is not None and lon_elem is not None:
                        lat = float(lat_elem.text)
                        lon = float(lon_elem.text)
                        elev = float(elev_elem.text) if elev_elem is not None else 0.0
                        
                        distance = locations2degrees(event_latitude, event_longitude, lat, lon)
                        if distance <= max_distance:
                            azimuth = sc_math.delazi(event_latitude, event_longitude, lat, lon)[1]
                            
                            stations.append({
                                "code": sta_code,
                                "network": net_code,
                                "latitude": lat,
                                "longitude": lon,
                                "elevation": elev,
                                "distance": distance,
                                "azimuth": azimuth
                            })
        
        logger.info(f"Loaded {len(stations)} stations from inventory file within {max_distance} degrees")
        if stations:
            logger.info(f"Station distances range from {min([s['distance'] for s in stations]):.2f} to {max([s['distance'] for s in stations]):.2f} degrees")
        else:
            logger.warning("No stations found in inventory file within the specified distance")
        
        return sorted(stations, key=lambda x: x['distance'])
    except Exception as e:
        logger.error(f"Error loading stations from inventory: {e}")
        return []

def load_stations_from_fdsnws(config: configparser.ConfigParser, event_latitude: float, event_longitude: float, 
                            configured_stations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Load station information from FDSNWS service.
    
    Args:
        config: Configuration parameters
        event_latitude: Event latitude in degrees
        event_longitude: Event longitude in degrees
        configured_stations: List of configured stations from parse_config_xml
        
    Returns:
        List of station dictionaries with coordinates and calculated distances
    """
    fdsn_url = config.get('FDSN', 'url', fallback='http://localhost:8081/fdsnws/station/1/query')
    max_distance = float(config['QualityParameters']['maximum_distance'])
    params = {
        'channel': 'BH?,HH?,SH?,EH?',
        'latitude': event_latitude,
        'longitude': event_longitude,
        'maxradius': max_distance,
        'level': 'channel',
        'format': 'text',
        'nodata': '404'
    }
    try:
        logger.info(f"Fetching station data from FDSNWS: {fdsn_url}")
        response = requests.get(fdsn_url, params=params)
        response.raise_for_status()
        csv_reader = csv.reader(StringIO(response.text), delimiter='|')
        next(csv_reader)  # Skip header
        stations = []
        for row in csv_reader:
            network, station, _, _, latitude, longitude, elevation = row[:7]
            # Check if the station is in the configured list
            if any(s['network'] == network and s['station'] == station for s in configured_stations):
                distance = locations2degrees(event_latitude, event_longitude, float(latitude), float(longitude))
                if distance <= max_distance:
                    azimuth = sc_math.delazi(event_latitude, event_longitude, float(latitude), float(longitude))[1]
                    stations.append({
                        "code": station,
                        "network": network,
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                        "elevation": float(elevation),
                        "distance": distance,
                        "azimuth": azimuth
                    })
        logger.info(f"Loaded {len(stations)} configured stations within {max_distance} degrees from FDSN web service")
        if stations:
            logger.info(f"Station distances range from {min([s['distance'] for s in stations]):.2f} to {max([s['distance'] for s in stations]):.2f} degrees")
        else:
            logger.warning("No stations were loaded from FDSNWS. Check your station inventory and FDSN web service.")
        return sorted(stations, key=lambda x: x['distance'])
    except requests.RequestException as e:
        logger.error(f"Error fetching data from FDSN web service: {e}")
        return []

def load_stations(config: configparser.ConfigParser, event_latitude: float, event_longitude: float, 
                configured_stations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Load station information, first from inventory file, then FDSNWS if needed.
    
    Args:
        config: Configuration parameters
        event_latitude: Event latitude in degrees
        event_longitude: Event longitude in degrees
        configured_stations: List of configured stations from parse_config_xml
        
    Returns:
        List of station dictionaries with coordinates and calculated distances
    """
    # First try to load stations from inventory file
    inventory_path = config.get('Inventory', 'path', fallback='inventory.xml')
    max_distance = float(config['QualityParameters']['maximum_distance'])
    
    logger.info(f"Attempting to load stations from inventory file: {inventory_path}")
    stations = load_stations_from_inventory(inventory_path, event_latitude, event_longitude, max_distance)
    
    # If inventory file doesn't provide enough stations, try FDSNWS as fallback
    if not stations and config.get('FDSN', 'url', fallback=''):
        logger.info("No stations loaded from inventory, attempting to use FDSNWS")
        stations = load_stations_from_fdsnws(config, event_latitude, event_longitude, configured_stations)
        
    if not stations:
        logger.warning("No stations could be loaded from inventory or FDSNWS")
        
    return stations

def generate_event_id(agency_id_lowercase: str, event_time: datetime.datetime) -> str:
    """
    Generate a unique event ID based on agency and time.
    
    Args:
        agency_id_lowercase: Lowercase agency ID
        event_time: Event time as datetime object
        
    Returns:
        Generated event ID string
    """
    current_year = event_time.year
    year_start = datetime.datetime(current_year, 1, 1)
    year_fraction = (event_time - year_start).total_seconds() / (366 * 24 * 60 * 60)
    letter_value = int(year_fraction * (26**6))
    letters = ''.join(string.ascii_lowercase[(letter_value // (26**i)) % 26] for i in range(5, -1, -1))
    return f"{agency_id_lowercase}{current_year}{letters}"

def seiscomp_time_to_datetime(sc_time: core.Time) -> datetime.datetime:
    """
    Convert SeisComp time to Python datetime.
    
    Args:
        sc_time: SeisComp time object
        
    Returns:
        Python datetime object
    """
    return datetime.datetime.strptime(sc_time.toString("%Y-%m-%d %H:%M:%S.%f"), "%Y-%m-%d %H:%M:%S.%f")

def create_pick(origin: datamodel.Origin, station: Dict[str, Any], distance_km: float, phase: str, 
              config: configparser.ConfigParser, depth_km: float) -> Optional[datamodel.Pick]:
    """
    Create a seismic phase pick for a station.
    
    Args:
        origin: Origin object
        station: Station dictionary
        distance_km: Distance in kilometers
        phase: Phase code (e.g., 'P', 'S')
        config: Configuration parameters
        depth_km: Event depth in kilometers
        
    Returns:
        Created Pick object or None if creation fails
    """
    pick = datamodel.Pick.Create()
    model = TauPyModel(model="iasp91")
    
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth_km,
                                          distance_in_degree=station['distance'],
                                          phase_list=[phase])
    except Exception as e:
        logger.error(f"Error in TauPyModel for station {station['code']}, phase {phase}: {str(e)}")
        return None
    
    if not arrivals:
        logger.warning(f"No arrivals found for station {station['code']}, phase {phase}")
        return None
    
    travel_time = arrivals[0].time
    pick_time = origin.time().value() + core.TimeSpan(travel_time)
    pick_time += core.TimeSpan(random.gauss(0, float(config['Noise']['pick_time_std'])))
    pick.setTime(datamodel.TimeQuantity(pick_time))
    pick.setPhaseHint(datamodel.Phase(phase))
    pick.setEvaluationMode(datamodel.AUTOMATIC)
    pick.setEvaluationStatus(datamodel.PRELIMINARY)
    
    # Assign correct channel codes based on phase
    if phase == 'P':
        channel_code = random.choice(['BHZ', 'HHZ', 'SHZ'])
    elif phase == 'S':
        channel_code = random.choice(['BHN', 'BHE', 'HHN', 'HHE', 'SHN', 'SHE'])
    else:
        channel_code = 'BHZ'  # Default to BHZ for unknown phases
    
    waveform_id = datamodel.WaveformStreamID(station['network'], station['code'], "00", channel_code, "")
    pick.setWaveformID(waveform_id)
    creation_info = datamodel.CreationInfo()
    creation_info.setAgencyID(config['Agency']['id'])
    creation_info.setAuthor(f"AutoPicker@{config['Agency']['id']}")
    creation_info.setCreationTime(core.Time.GMT())
    pick.setCreationInfo(creation_info)
    return pick

def create_arrival(pick: datamodel.Pick, distance_deg: float, azimuth: float, phase: str, 
                 config: configparser.ConfigParser, theoretical_time: core.Time) -> datamodel.Arrival:
    """
    Create an arrival for a pick.
    
    Args:
        pick: Pick object
        distance_deg: Distance in degrees
        azimuth: Azimuth in degrees
        phase: Phase code
        config: Configuration parameters
        theoretical_time: Theoretical arrival time
        
    Returns:
        Created Arrival object
    """
    arrival = datamodel.Arrival()
    arrival.setPickID(pick.publicID())
    arrival.setPhase(datamodel.Phase(phase))
    arrival.setDistance(distance_deg)
    arrival.setAzimuth(azimuth)
    time_residual = pick.time().value().seconds() - theoretical_time.seconds()
    arrival.setTimeResidual(time_residual)
    arrival.setTimeUsed(True)
    arrival.setWeight(1.0)
    return arrival

def create_picks_and_arrivals(origin: datamodel.Origin, stations: List[Dict[str, Any]], 
                            config: configparser.ConfigParser) -> Tuple[List[datamodel.Pick], List[datamodel.Arrival], List[str]]:
    """
    Create picks and arrivals for an origin using station data.
    
    Args:
        origin: Origin object
        stations: List of station dictionaries
        config: Configuration parameters
        
    Returns:
        Tuple containing lists of picks, arrivals, and used station codes
    """
    picks = []
    arrivals = []
    used_stations = []
    origin_lat, origin_lon = origin.latitude().value(), origin.longitude().value()
    s_wave_cutoff = float(config['Phases']['s_wave_cutoff'])
    model = TauPyModel(model="iasp91")
    
    # Convert depth from meters to kilometers
    depth_km = origin.depth().value()
    logger.debug(f"Origin depth = {depth_km:.2f} km")
    
    max_distance = float(config['QualityParameters']['maximum_distance'])
    min_distance = float(config['Inventory']['min_distance'])
    
    logger.info(f"Processing {len(stations)} stations")
    for idx, station in enumerate(stations):
        if idx % 10 == 0:
            logger.debug(f"Processing station {idx+1}/{len(stations)}")
        distance_deg = station['distance']
        azimuth = station['azimuth']
        
        if not (min_distance <= distance_deg <= max_distance):
            continue
        
        if station['code'] in used_stations:
            continue
        
        used_stations.append(station['code'])
        phases_to_pick = ['P', 'S'] if distance_deg <= s_wave_cutoff else ['P']
        
        for phase in phases_to_pick:
            logger.debug(f"Creating pick for station {station['code']}, phase {phase}, origin depth = {depth_km:.2f} km")
            pick = create_pick(origin, station, degrees2kilometers(distance_deg), phase, config, depth_km)
            if pick:
                picks.append(pick)
                theoretical_arrivals = model.get_travel_times(source_depth_in_km=depth_km,
                                                           distance_in_degree=distance_deg,
                                                           phase_list=[phase])
                if theoretical_arrivals:
                    theoretical_time = origin.time().value() + core.TimeSpan(theoretical_arrivals[0].time)
                    arrival = create_arrival(pick, distance_deg, azimuth, phase, config, theoretical_time)
                    arrivals.append(arrival)
    
    logger.info(f"Created {len(picks)} picks and {len(arrivals)} arrivals for {len(used_stations)} stations")
    return picks, arrivals, used_stations

def create_station_magnitudes(origin: datamodel.Origin, magnitude: datamodel.Magnitude, stations: List[Dict[str, Any]], 
                            config: configparser.ConfigParser, used_stations: List[str]) -> List[datamodel.StationMagnitude]:
    """
    Create station magnitudes for a network magnitude.
    
    Args:
        origin: Origin object
        magnitude: Network magnitude object
        stations: List of station dictionaries
        config: Configuration parameters
        used_stations: List of station codes used for picks
        
    Returns:
        List of created StationMagnitude objects
    """
    station_magnitudes = []
    used_stations_set = set(used_stations)
    logger.info(f"Creating station magnitudes for magnitude type {magnitude.type()}")
    
    max_distance = float(config['QualityParameters']['maximum_distance'])
    stations = filter_stations_by_distance(stations, max_distance)
    
    # Shuffle the stations to randomize selection
    random.shuffle(stations)
    
    for station in stations:
        station_key = f"{station['network']}.{station['code']}"
        if station['code'] not in used_stations_set:
            continue
        
        sta_mag = datamodel.StationMagnitude.Create()
        
        mag_value = magnitude.magnitude().value() + random.gauss(0, float(config['Noise']['station_magnitude_std']))
        sta_mag.setMagnitude(datamodel.RealQuantity(mag_value, float(config['Noise']['station_magnitude_std'])))
        
        sta_mag.setType(magnitude.type())  # Use the exact magnitude type from the parent magnitude
        sta_mag.setOriginID(origin.publicID())
        sta_mag.setMethodID("average")
        
        waveform_id = datamodel.WaveformStreamID()
        waveform_id.setNetworkCode(station['network'])
        waveform_id.setStationCode(station['code'])
        waveform_id.setChannelCode("BHZ")
        waveform_id.setLocationCode("00")
        sta_mag.setWaveformID(waveform_id)
        
        contrib = datamodel.StationMagnitudeContribution()
        contrib.setStationMagnitudeID(sta_mag.publicID())
        contrib.setResidual(random.gauss(0, 0.1))
        contrib.setWeight(1.0)
        
        magnitude.add(contrib)
        station_magnitudes.append(sta_mag)
        
        logger.debug(f"Created station magnitude {sta_mag.magnitude().value():.2f} for station {station['network']}.{station['code']}")
        
        # Remove the station from the set to ensure it's not used again
        used_stations_set.remove(station['code'])
        
        # Break if we've used all available stations
        if not used_stations_set:
            break
    
    logger.info(f"Created {len(station_magnitudes)} station magnitudes")
    return station_magnitudes

def create_multiple_origins(config: configparser.ConfigParser, event_time: core.Time, stations: List[Dict[str, Any]], 
                          event_depth_km: float) -> Tuple[List[datamodel.Origin], List[datamodel.Pick], List[List[str]]]:
    """
    Create multiple origins for an event with progressive refinement.
    
    Args:
        config: Configuration parameters
        event_time: Event time
        stations: List of station dictionaries
        event_depth_km: Event depth in kilometers
        
    Returns:
        Tuple containing lists of Origins, Picks, and lists of used station codes per origin
    """
    origins = []
    all_picks = []
    all_used_stations = []
    num_origins = int(config['MultipleOrigins']['number_of_origins'])
    creation_time_increment = float(config['MultipleOrigins']['creation_time_increment'])
    base_creation_time = core.Time.GMT()

    lat = float(config['Event']['latitude'])
    lon = float(config['Event']['longitude'])
    depth_km = event_depth_km  # Use the provided event depth
    depth_uncertainty_km = float(config['Uncertainties']['origin_depth'])

    initial_station_count = int(config['MultipleOrigins']['initial_station_count'])
    station_increase_per_origin = int(config['MultipleOrigins']['station_increase_per_origin'])

    max_distance = float(config['QualityParameters']['maximum_distance'])
    min_distance = float(config['Inventory'].get('min_distance', 0))

    # Sort all stations by distance once
    sorted_stations = sorted(stations, key=lambda x: x['distance'])

    logger.info(f"Creating {num_origins} origins")
    logger.info(f"Total available stations: {len(stations)}")
    logger.info(f"Stations within distance range {min_distance}-{max_distance}: {len([s for s in stations if min_distance <= s['distance'] <= max_distance])}")

    if not stations:
        logger.warning("No stations available. Check your station inventory and FDSN web service.")
        return [], [], []

    for i in range(num_origins):
        logger.info(f"\nCreating origin {i+1}/{num_origins}")
        origin = datamodel.Origin.Create()
        
        lat_value = lat + np.random.normal(0, float(config['Uncertainties']['origin_latitude']))
        lat_uncertainty = float(config['Uncertainties']['origin_latitude']) * (1 - i/num_origins)
        origin.setLatitude(datamodel.RealQuantity(lat_value, lat_uncertainty))
        
        lon_value = lon + np.random.normal(0, float(config['Uncertainties']['origin_longitude']))
        lon_uncertainty = float(config['Uncertainties']['origin_longitude']) * (1 - i/num_origins)
        origin.setLongitude(datamodel.RealQuantity(lon_value, lon_uncertainty))
        
        # Correct depth handling
        depth_value = depth_km + np.random.normal(0, depth_uncertainty_km)
        depth_uncertainty = depth_uncertainty_km * (1 - i/num_origins)
        origin.setDepth(datamodel.RealQuantity(depth_value, depth_uncertainty))  # Convert to meters for SeisComp
        
        origin.setTime(datamodel.TimeQuantity(event_time))
        creation_time = base_creation_time + core.TimeSpan(i * creation_time_increment)
        creation_info = datamodel.CreationInfo()
        creation_info.setCreationTime(creation_time)
        creation_info.setAgencyID(config['Agency']['id'])
        creation_info.setAuthor(f"OriginLocator@{config['Agency']['id']}")
        origin.setCreationInfo(creation_info)
        origin.setEvaluationMode(datamodel.AUTOMATIC)
        origin.setEvaluationStatus(datamodel.PRELIMINARY)
        
        logger.info(f"Origin depth: {depth_value:.2f} km")
        
        # Calculate the number of stations to use for this origin
        num_stations = min(initial_station_count + i * station_increase_per_origin, len(stations))
        
        # Select stations, prioritizing closer ones but also including some distant ones
        close_station_ratio = 0.7  # 70% of stations will be the closest ones
        close_station_count = int(num_stations * close_station_ratio)
        distant_station_count = num_stations - close_station_count

        close_stations = sorted_stations[:close_station_count]
        distant_stations = sorted_stations[close_station_count:]
        
        # Randomly select distant stations
        if distant_station_count > 0 and distant_stations:
            distant_stations = random.sample(distant_stations, min(distant_station_count, len(distant_stations)))
        
        stations_to_use = close_stations + distant_stations
        
        # Filter stations by distance range
        stations_to_use = [s for s in stations_to_use if min_distance <= s['distance'] <= max_distance]
        
        logger.info(f"Stations available for this origin: {len(stations_to_use)}")
        if stations_to_use:
            logger.info(f"Distance range of used stations: {min([s['distance'] for s in stations_to_use]):.2f} to {max([s['distance'] for s in stations_to_use]):.2f} degrees")
        
        if not stations_to_use:
            logger.warning(f"Warning: No stations available for origin {i+1}. Skipping this origin.")
            continue
        
        picks, arrivals, used_stations = create_picks_and_arrivals(origin, stations_to_use, config)
        logger.info(f"Created {len(picks)} picks and {len(arrivals)} arrivals")
        all_picks.extend(picks)
        for arrival in arrivals:
            origin.add(arrival)
        
        logger.info("Setting origin quality")
        quality = datamodel.OriginQuality()
        quality.setAssociatedPhaseCount(len(arrivals))
        quality.setUsedPhaseCount(len(arrivals))
        quality.setAssociatedStationCount(len(used_stations))
        quality.setUsedStationCount(len(used_stations))
        quality.setDepthPhaseCount(len([a for a in arrivals if a.phase().code() in ["pP", "sP"]]))
        quality.setStandardError(float(config['QualityParameters']['standard_error']))
        quality.setAzimuthalGap(calculate_azimuthal_gap(stations_to_use, used_stations))
        quality.setSecondaryAzimuthalGap(float(config['QualityParameters']['secondary_azimuthal_gap']))
        quality.setGroundTruthLevel(config['QualityParameters']['ground_truth_level'])
        
        if stations_to_use:
            quality.setMaximumDistance(max(s['distance'] for s in stations_to_use))
            quality.setMinimumDistance(min(s['distance'] for s in stations_to_use))
            quality.setMedianDistance(np.median([s['distance'] for s in stations_to_use]))
        else:
            logger.warning("No stations available for setting distance parameters in origin quality.")
        
        origin.setQuality(quality)
        
        origins.append(origin)
        all_used_stations.append(used_stations)
        logger.info(f"Finished creating origin {i+1}/{num_origins} using {len(stations_to_use)} stations")

    logger.info(f"\nCreated {len(origins)} origins with {len(all_picks)} total picks")
    return origins, all_picks, all_used_stations

def create_magnitudes(origin: datamodel.Origin, stations: List[Dict[str, Any]], 
                    config: configparser.ConfigParser, focal_mechanisms: List[datamodel.FocalMechanism], 
                    used_stations: List[str]) -> List[datamodel.Magnitude]:
    """
    Create network magnitudes for an origin.
    
    Args:
        origin: Origin object
        stations: List of station dictionaries
        config: Configuration parameters
        focal_mechanisms: List of focal mechanism objects
        used_stations: List of station codes used for picks
        
    Returns:
        List of created Magnitude objects
    """
    mags = []
    magnitude_config = config['Magnitudes']
    
    # Filter stations to use only those that were used for the origin
    stations_to_use = [s for s in stations if s['code'] in used_stations]
    
    for mag_type, mag_value in magnitude_config.items():
        if mag_type != 'Mww':  # Handle regular magnitudes
            mag = datamodel.Magnitude.Create()
            
            mag_value_with_noise = float(mag_value) + np.random.normal(0, float(config['Noise']['station_magnitude_std']))
            mag.setMagnitude(datamodel.RealQuantity(mag_value_with_noise, float(config['Noise']['station_magnitude_std'])))
            mag.setType(mag_type)  # Use the exact magnitude type from config
            mag.setOriginID(origin.publicID())
            mag.setStationCount(len(stations_to_use))
            
            mags.append(mag)

    return mags

def create_mww_magnitudes(origins: List[datamodel.Origin], config: configparser.ConfigParser
                       ) -> Tuple[List[datamodel.Magnitude], List[datamodel.Origin]]:
    """
    Create Mww magnitudes and centroid origins.
    
    Args:
        origins: List of origin objects
        config: Configuration parameters
        
    Returns:
        Tuple containing lists of Mww magnitudes and centroid origins
    """
    mww_magnitudes = []
    centroid_origins = []
    mww_values = [float(val.strip()) for val in config['Magnitudes']['Mww'].split(',')]
    
    if len(mww_values) != 3:
        logger.warning("Expected 3 Mww values in config. Using default values.")
        mww_values = [8.7, 9.1, 9.3]  # Default values
    
    mww_origins = origins[-3:]  # Use the last three origins for Mww

    for origin, mww_value in zip(mww_origins, mww_values):
        # Create Mww magnitude
        mww_mag = datamodel.Magnitude.Create()
        mww_mag.setMagnitude(datamodel.RealQuantity(mww_value, 0.1))
        mww_mag.setType("Mww")
        mww_mag.setOriginID(origin.publicID())
        mww_mag.setMethodID("wphase")
        mww_mag.setStationCount(origin.quality().usedStationCount())
        
        # Create centroid origin
        centroid = datamodel.Origin.Create()
        centroid_lat = origin.latitude().value() + random.uniform(-0.1, 0.1)
        centroid_lon = origin.longitude().value() + random.uniform(-0.1, 0.1)
        centroid_depth = origin.depth().value() + random.uniform(-5000, 5000)  # In meters
        centroid_time = origin.time().value() + core.TimeSpan(random.uniform(-5, 5))
        
        centroid.setLatitude(datamodel.RealQuantity(centroid_lat, 0.05))
        centroid.setLongitude(datamodel.RealQuantity(centroid_lon, 0.05))
        centroid.setDepth(datamodel.RealQuantity(centroid_depth, 2500))  # In meters
        centroid.setTime(datamodel.TimeQuantity(centroid_time))
        
        # Set centroid as the derived origin for the Mww magnitude
        mww_mag.setDerivedOriginID(centroid.publicID())
        
        # Add centroid information as a comment to Mww magnitude
        centroid_comment = datamodel.Comment()
        centroid_comment.setText(f"Centroid: Lat={centroid_lat:.4f}, Lon={centroid_lon:.4f}, Depth={centroid_depth/1000:.2f} km, Time={centroid_time.toString('%Y-%m-%d %H:%M:%S.%f')}")
        mww_mag.add(centroid_comment)
        
        mww_magnitudes.append(mww_mag)
        centroid_origins.append(centroid)

    return mww_magnitudes, centroid_origins

def create_focal_mechanisms(config: configparser.ConfigParser, origins: List[datamodel.Origin], event_id: str
                         ) -> Tuple[List[datamodel.FocalMechanism], List[datamodel.Magnitude], List[datamodel.Origin]]:
    """
    Create focal mechanisms, Mww magnitudes, and centroid origins.
    
    Args:
        config: Configuration parameters
        origins: List of origin objects
        event_id: Event ID string
        
    Returns:
        Tuple containing lists of focal mechanisms, Mww magnitudes, and centroid origins
    """
    focal_mechanisms = []
    mww_magnitudes = []
    centroid_origins = []
    fm_count = int(config['FocalMechanism'].get('count', 1))
    agency_id = config['Agency']['id']
    
    mww_values = [float(val.strip()) for val in config['Magnitudes']['Mww'].split(',')]
    if len(mww_values) != fm_count:
        logger.warning(f"Expected {fm_count} Mww values in config. Using default values.")
        mww_values = [9.0] * fm_count  # Default value

    for i in range(fm_count):
        fm = datamodel.FocalMechanism.Create()
        fm.setPublicID(f"{agency_id}/focalmechanism/{event_id}/{i}")
        fm.setMethodID("wphase")  # Set method ID to 'wphase'
        
        fm.setTriggeringOriginID(origins[-1].publicID())
        
        # Create nodal planes
        np = datamodel.NodalPlanes()
        np1 = datamodel.NodalPlane()
        np1.setStrike(datamodel.RealQuantity(float(config['FocalMechanism'][f'strike1_{i+1}']), 5.0))
        np1.setDip(datamodel.RealQuantity(float(config['FocalMechanism'][f'dip1_{i+1}']), 5.0))
        np1.setRake(datamodel.RealQuantity(float(config['FocalMechanism'][f'rake1_{i+1}']), 5.0))
        np.setNodalPlane1(np1)
        np2 = datamodel.NodalPlane()
        np2.setStrike(datamodel.RealQuantity(float(config['FocalMechanism'][f'strike2_{i+1}']), 5.0))
        np2.setDip(datamodel.RealQuantity(float(config['FocalMechanism'][f'dip2_{i+1}']), 5.0))
        np2.setRake(datamodel.RealQuantity(float(config['FocalMechanism'][f'rake2_{i+1}']), 5.0))
        np.setNodalPlane2(np2)
        fm.setNodalPlanes(np)
        
        # Create moment tensor, Mww magnitude, and centroid
        mt, mag_mww, centroid = create_moment_tensor(config, origins[-1], event_id, i, mww_values[i])
        fm.add(mt)  # Add moment tensor to focal mechanism

        # Populate derived origin (centroid) properties
        if origins[-1].quality():
            quality = datamodel.OriginQuality()
            quality.setUsedPhaseCount(origins[-1].quality().usedPhaseCount())
            quality.setAzimuthalGap(origins[-1].quality().azimuthalGap())
            quality.setUsedStationCount(origins[-1].quality().usedStationCount())
            quality.setAssociatedStationCount(origins[-1].quality().associatedStationCount())
            quality.setAssociatedPhaseCount(origins[-1].quality().associatedPhaseCount())
            centroid.setQuality(quality)
        
        # Calculate and set misfit (this is a placeholder, adjust as needed)
        misfit = random.uniform(0.1, 0.5)
        fm.setMisfit(misfit)
        
        mww_magnitudes.append(mag_mww)
        centroid_origins.append(centroid)
        
        # Set evaluation parameters
        fm.setEvaluationMode(datamodel.AUTOMATIC)
        fm.setEvaluationStatus(datamodel.CONFIRMED)
        
        # Set creation info
        creation_info = datamodel.CreationInfo()
        creation_info.setAgencyID(agency_id)
        creation_info.setAuthor(f"AutoFM@{agency_id}")
        creation_info.setCreationTime(core.Time.GMT())
        fm.setCreationInfo(creation_info)
        
        # Explicitly link focal mechanism, Mww, and centroid
        fm_comment = datamodel.Comment()
        fm_comment.setText(f"Associated Mww ID: {mag_mww.publicID()}, Centroid ID: {centroid.publicID()}")
        fm.add(fm_comment)

        mww_comment = datamodel.Comment()
        mww_comment.setText(f"Associated Focal Mechanism ID: {fm.publicID()}, Centroid ID: {centroid.publicID()}")
        mag_mww.add(mww_comment)

        centroid_comment = datamodel.Comment()
        centroid_comment.setText(f"Associated Focal Mechanism ID: {fm.publicID()}, Mww ID: {mag_mww.publicID()}")
        centroid.add(centroid_comment)
        
        focal_mechanisms.append(fm)
    
    return focal_mechanisms, mww_magnitudes, centroid_origins

def create_moment_tensor(config: configparser.ConfigParser, origin: datamodel.Origin, event_id: str, 
                       index: int, mw: float) -> Tuple[datamodel.MomentTensor, datamodel.Magnitude, datamodel.Origin]:
    """
    Create a moment tensor and associated objects.
    
    Args:
        config: Configuration parameters
        origin: Origin object
        event_id: Event ID string
        index: Index for the focal mechanism
        mw: Moment magnitude value
        
    Returns:
        Tuple containing the moment tensor, Mww magnitude, and centroid origin
    """
    mt = datamodel.MomentTensor.Create()
    agency_id = config['Agency']['id']
    
    # Calculate scalar moment
    scalar_moment = 10 ** (1.5 * mw + 9.1)
    mt.setScalarMoment(datamodel.RealQuantity(scalar_moment))
    
    # Calculate tensor components
    strike = float(config['FocalMechanism'][f'strike1_{index+1}'])
    dip = float(config['FocalMechanism'][f'dip1_{index+1}'])
    rake = float(config['FocalMechanism'][f'rake1_{index+1}'])
    tensor = calculate_tensor_components(scalar_moment, strike, dip, rake)
    mt.setTensor(tensor)
    
    # Set derived parameters
    mt.setDoubleCouple(0.8 + random.uniform(-0.1, 0.1))
    mt.setClvd(0.1 + random.uniform(-0.05, 0.05))
    
    # Create centroid origin
    centroid = datamodel.Origin.Create()
    centroid_lat = origin.latitude().value() + random.uniform(-0.1, 0.1)
    centroid_lon = origin.longitude().value() + random.uniform(-0.1, 0.1)
    centroid_depth_km = float(config['Event']['depth'])
    centroid_time = origin.time().value() + core.TimeSpan(random.uniform(-5, 5))
    
    centroid.setLatitude(datamodel.RealQuantity(centroid_lat, 0.05))
    centroid.setLongitude(datamodel.RealQuantity(centroid_lon, 0.05))
    centroid.setDepth(datamodel.RealQuantity(centroid_depth_km , 2500))  # Convert to meters
    centroid.setTime(datamodel.TimeQuantity(centroid_time))
    
    # Set the centroid origin ID for the moment tensor
    mt.setDerivedOriginID(centroid.publicID())
    
    # Create and link Mww magnitude
    mag_mww = datamodel.Magnitude.Create()
    mag_mww.setMagnitude(datamodel.RealQuantity(mw, 0.1))
    mag_mww.setType("Mww")
    mag_mww.setOriginID(centroid.publicID())  # Link to centroid origin
    mag_mww.setMethodID("wphase")
    mag_mww.setStationCount(origin.quality().usedStationCount())
    mag_mww.setCreationInfo(origin.creationInfo())
    
    # Add Mww magnitude to centroid
    centroid.add(mag_mww)
    
    # Add strike, dip, rake to Mww
    comment = datamodel.Comment()
    comment.setText(f"Strike={strike:.1f}, Dip={dip:.1f}, Rake={rake:.1f}")
    mag_mww.add(comment)
    
    return mt, mag_mww, centroid

def calculate_tensor_components(scalar_moment: float, strike: float, dip: float, rake: float) -> datamodel.Tensor:
    """
    Calculate moment tensor components from scalar moment and focal mechanism.
    
    Args:
        scalar_moment: Scalar moment value
        strike: Strike angle in degrees
        dip: Dip angle in degrees
        rake: Rake angle in degrees
        
    Returns:
        Tensor object with calculated components
    """
    s, d, r = np.radians([strike, dip, rake])
    
    mrr = scalar_moment * (np.sin(2*d) * np.sin(r))
    mtt = -scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
    mpp = scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
    mrt = -scalar_moment * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
    mrp = scalar_moment * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
    mtp = -scalar_moment * (np.sin(d) * np.cos(r) * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
    
    tensor = datamodel.Tensor()
    tensor.setMrr(datamodel.RealQuantity(mrr))
    tensor.setMtt(datamodel.RealQuantity(mtt))
    tensor.setMpp(datamodel.RealQuantity(mpp))
    tensor.setMrt(datamodel.RealQuantity(mrt))
    tensor.setMrp(datamodel.RealQuantity(mrp))
    tensor.setMtp(datamodel.RealQuantity(mtp))
    
    return tensor

def calculate_azimuthal_gap(stations: List[Dict[str, Any]], used_stations: List[str]) -> float:
    """
    Calculate the azimuthal gap for an origin.
    
    Args:
        stations: List of station dictionaries
        used_stations: List of station codes used for picks
        
    Returns:
        Azimuthal gap in degrees
    """
    if not used_stations:
        return 360.0  # Full gap if no stations
    
    station_dict = {s['code']: s['azimuth'] for s in stations}
    azimuths = sorted([station_dict[code] for code in used_stations if code in station_dict])
    
    if not azimuths:
        return 360.0  # Full gap if no valid azimuths
    
    gaps = np.diff(azimuths)
    gaps = np.append(gaps, 360 + azimuths[0] - azimuths[-1])
    return np.max(gaps)

def create_synthetic_event(config: configparser.ConfigParser, seiscomp_config_file: str) -> datamodel.EventParameters:
    """
    Create a synthetic seismic event with all components.
    
    Args:
        config: Configuration parameters
        seiscomp_config_file: Path to the SeisComp configuration file
        
    Returns:
        EventParameters object containing the created event
    """
    ep = datamodel.EventParameters()

    agency_id = config['Agency']['id']
    agency_id_lowercase = config['Agency']['id_lowercase']

    if config['Event']['time'].lower() == 'now':
        event_time = core.Time.GMT()
    else:
        event_time = core.Time.FromString(config['Event']['time'], "%FT%T%Z")

    py_event_time = seiscomp_time_to_datetime(event_time)

    event_id = generate_event_id(agency_id_lowercase, py_event_time)

    event = datamodel.Event.Create(event_id)
    ep.add(event)
    event.setType(getattr(datamodel, config['Event']['type']))

    creation_info = datamodel.CreationInfo()
    creation_info.setAuthor(f"AutoDetector@{agency_id}")
    creation_info.setAgencyID(agency_id)
    creation_info.setCreationTime(core.Time.GMT())
    event.setCreationInfo(creation_info)

    event_latitude = float(config['Event']['latitude'])
    event_longitude = float(config['Event']['longitude'])
    event_depth_km = float(config['Event']['depth'])

    try:
        configured_stations = parse_config_xml(seiscomp_config_file)
        global_stations = load_stations(config, event_latitude, event_longitude, configured_stations)
        logger.info(f"Loaded {len(global_stations)} configured stations for event generation")

        if not global_stations:
            logger.error("No stations could be loaded. Cannot proceed with event generation.")
            return ep

        logger.info("Creating multiple origins")
        origins, picks, all_used_stations = create_multiple_origins(config, event_time, global_stations, event_depth_km)
        logger.info(f"Created {len(origins)} origins and {len(picks)} picks")

        if not origins:
            logger.warning("No origins were created. Cannot proceed with event generation.")
            return ep

        logger.info("Creating focal mechanisms")
        focal_mechanisms, mww_magnitudes, centroid_origins = create_focal_mechanisms(config, origins, event_id)
        logger.info(f"Created {len(focal_mechanisms)} focal mechanisms with Mww magnitudes and centroids")
        
        logger.info("Adding picks to event parameters")
        for pick in picks:
            ep.add(pick)

        logger.info("Adding origins, magnitudes, and arrivals to event parameters")
        for origin, used_stations in zip(origins, all_used_stations):
            ep.add(origin)
            event.add(datamodel.OriginReference(origin.publicID()))

            logger.info(f"Creating magnitudes for origin {origin.publicID()}")
            mags = create_magnitudes(origin, global_stations, config, focal_mechanisms, used_stations)
            for mag in mags:
                origin.add(mag)
                logger.info(f"Created magnitude: {mag.type()} = {mag.magnitude().value():.2f}")

            logger.info(f"Creating station magnitudes for origin {origin.publicID()}")
            used_stations_copy = used_stations.copy()
            for mag in mags:
                station_magnitudes = create_station_magnitudes(origin, mag, global_stations, config, used_stations_copy)
                for sta_mag in station_magnitudes:
                    origin.add(sta_mag)
            logger.info(f"Created station magnitudes for origin {origin.publicID()}")

        if focal_mechanisms:
            logger.info("Adding focal mechanisms, Mww magnitudes, and centroids to event parameters")
            for fm, mww_mag, centroid in zip(focal_mechanisms, mww_magnitudes, centroid_origins):
                ep.add(fm)
                ep.add(centroid)
                
                # Add Mww magnitude to the centroid origin
                centroid.add(mww_mag)
                
                # Create a MomentTensor object if it doesn't exist
                if fm.momentTensorCount() == 0:
                    mt = datamodel.MomentTensor.Create()
                    fm.add(mt)
                else:
                    mt = fm.momentTensor(0)
                
                # Associate the Mww magnitude with the MomentTensor
                mt.setMomentMagnitudeID(mww_mag.publicID())
                
                # Set the derived origin (centroid) for the moment tensor
                mt.setDerivedOriginID(centroid.publicID())
                
                # Add references to the event
                event.add(datamodel.FocalMechanismReference(fm.publicID()))
                event.add(datamodel.OriginReference(centroid.publicID()))
                
                # Instead of MagnitudeReference, we set the preferred magnitude ID
                event.setPreferredMagnitudeID(mww_mag.publicID())
                
                logger.info(f"Added focal mechanism {fm.publicID()} with Mww {mww_mag.magnitude().value():.2f}")
                logger.info(f"  Centroid: Lat={centroid.latitude().value():.4f}, Lon={centroid.longitude().value():.4f}, Depth={centroid.depth().value()/1000:.2f} km")

        # Set preferred entities
        if origins:
            preferred_origin = origins[-1]
            event.setPreferredOriginID(preferred_origin.publicID())
        
        if mww_magnitudes:
            event.setPreferredMagnitudeID(mww_magnitudes[-1].publicID())
        
        if focal_mechanisms:
            event.setPreferredFocalMechanismID(focal_mechanisms[-1].publicID())

        logger.info(f"\nCreated event with ID: {event.publicID()}")
        logger.info(f"Preferred Origin ID: {event.preferredOriginID()}")
        logger.info(f"Preferred Magnitude ID: {event.preferredMagnitudeID()}")
        logger.info(f"Preferred Focal Mechanism ID: {event.preferredFocalMechanismID()}")

        # Print summary of the preferred origin
        preferred_origin = ep.findOrigin(event.preferredOriginID())
        if preferred_origin:
            logger.info("\nPreferred Origin Summary:")
            logger.info(f"  Time: {preferred_origin.time().value().toString('%Y-%m-%d %H:%M:%S.%f')}")
            logger.info(f"  Latitude: {preferred_origin.latitude().value():.4f} ± {preferred_origin.latitude().uncertainty():.4f}")
            logger.info(f"  Longitude: {preferred_origin.longitude().value():.4f} ± {preferred_origin.longitude().uncertainty():.4f}")
            logger.info(f"  Depth: {preferred_origin.depth().value()/1000:.2f} ± {preferred_origin.depth().uncertainty()/1000:.2f} km")
            logger.info(f"  Evaluation Mode: {preferred_origin.evaluationMode()}")
            logger.info(f"  Evaluation Status: {preferred_origin.evaluationStatus()}")
            
            logger.info("\n  Magnitudes:")
            for i in range(preferred_origin.magnitudeCount()):
                mag = preferred_origin.magnitude(i)
                logger.info(f"    {mag.type()}: {mag.magnitude().value():.2f} ± {mag.magnitude().uncertainty():.2f}")
                if mag.type() == "Mww":
                    for j in range(mag.commentCount()):
                        comment = mag.comment(j)
                        if comment.text().startswith("Mww solution"):
                            logger.info(f"      {comment.text()}")
    except Exception as e:
        logger.error(f"Error in event creation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    return ep

def write_to_xml(ep: datamodel.EventParameters, filename_prefix: str) -> bool:
    """
    Write event parameters to an XML file.
    
    Args:
        ep: EventParameters object
        filename_prefix: Prefix for the output file name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the event ID from the EventParameters object
        event_id = ep.event(0).publicID() if ep.eventCount() > 0 else "unknown"
        
        # Create a filename with the event ID
        filename = f"{filename_prefix}_{event_id}.xml"
        
        ar = io.XMLArchive()
        if not ar.create(filename):
            logger.error(f"Could not create file: {filename}")
            return False
        ar.setFormattedOutput(True)
        ar.writeObject(ep)
        ar.close()
        logger.info(f"Event has been written to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error writing to XML: {e}")
        return False

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Seismic Event Generator')
    parser.add_argument('config_file', type=str, help='Configuration file')
    parser.add_argument('seiscomp_config_file', type=str, help='SeisComp configuration file')
    parser.add_argument('--output', type=str, default="synthetic_event_seiscomp", help='Output file prefix')
    return parser.parse_args()

def main():
    """
    Main function to run the seismic event generator.
    """
    try:
        args = parse_args()
        
        # Use ConfigManager for better configuration handling
        config_manager = ConfigManager(args.config_file)
        config = config_manager.get_config()
        
        logger.info(f"Using configuration file: {args.config_file}")
        logger.info(f"Using SeisComp configuration file: {args.seiscomp_config_file}")
        
        ep = create_synthetic_event(config, args.seiscomp_config_file)
        logger.info("Successfully created synthetic event")

        # Use the specified or default prefix for the output file
        output_file_prefix = args.output
        if write_to_xml(ep, output_file_prefix):
            logger.info("Event has been written to file")
        else:
            logger.error("Failed to write event to file")

        # Print summary of generated event
        for i in range(ep.eventCount()):
            event = ep.event(i)
            logger.info(f"\nEvent {i+1} Summary:")
            logger.info(f"ID: {event.publicID()}")
            logger.info(f"Type: {event.type()}")
            logger.info(f"Creation Time: {event.creationInfo().creationTime().toString('%Y-%m-%d %H:%M:%S.%f')}")
            
            preferred_origin = ep.findOrigin(event.preferredOriginID())
            if preferred_origin:
                logger.info("\nPreferred Origin:")
                logger.info(f"  ID: {preferred_origin.publicID()}")
                logger.info(f"  Time: {preferred_origin.time().value().toString('%Y-%m-%d %H:%M:%S.%f')}")
                logger.info(f"  Latitude: {preferred_origin.latitude().value():.4f} ± {preferred_origin.latitude().uncertainty():.4f}")
                logger.info(f"  Longitude: {preferred_origin.longitude().value():.4f} ± {preferred_origin.longitude().uncertainty():.4f}")
                logger.info(f"  Depth: {preferred_origin.depth().value()/1000:.2f} ± {preferred_origin.depth().uncertainty()/1000:.2f} km")
                logger.info(f"  Evaluation Mode: {preferred_origin.evaluationMode()}")
                logger.info(f"  Evaluation Status: {preferred_origin.evaluationStatus()}")
                
                logger.info("\n  Magnitudes:")
                for j in range(preferred_origin.magnitudeCount()):
                    magnitude = preferred_origin.magnitude(j)
                    if magnitude.magnitude().uncertainty() is not None:
                        logger.info(f"    {magnitude.type()}: {magnitude.magnitude().value():.2f} ± {magnitude.magnitude().uncertainty():.2f}")
                    else:
                        logger.info(f"    {magnitude.type()}: {magnitude.magnitude().value():.2f} (uncertainty not set)")
            
            preferred_fm = ep.findFocalMechanism(event.preferredFocalMechanismID())
            if preferred_fm:
                logger.info("\nPreferred Focal Mechanism:")
                logger.info(f"  ID: {preferred_fm.publicID()}")
                logger.info(f"  Triggering Origin ID: {preferred_fm.triggeringOriginID()}")
                np = preferred_fm.nodalPlanes()
                if np:
                    np1 = np.nodalPlane1()
                    logger.info(f"  Nodal Plane 1: Strike {np1.strike().value():.1f}° ± {np1.strike().uncertainty():.1f}°, "
                          f"Dip {np1.dip().value():.1f}° ± {np1.dip().uncertainty():.1f}°, "
                          f"Rake {np1.rake().value():.1f}° ± {np1.rake().uncertainty():.1f}°")
                
                if preferred_fm.momentTensorCount() > 0:
                    mt = preferred_fm.momentTensor(0)
                    logger.info("\n  Moment Tensor:")
                    logger.info(f"    Derived Origin ID: {mt.derivedOriginID()}")
                    tensor = mt.tensor()
                    logger.info(f"    Scalar Moment: {mt.scalarMoment().value():.2e} Nm")
                    logger.info(f"    Double Couple: {mt.doubleCouple():.2f}")
                    logger.info(f"    CLVD: {mt.clvd():.2f}")
                    logger.info(f"    Moment Magnitude ID: {mt.momentMagnitudeID()}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Error details:")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()