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

def create_resource_id(agency_id, id_string):
    return f"smi:{agency_id.lower()}/{id_string}"

def parse_config_xml(config_file):
    tree = ET.parse(config_file)
    root = tree.getroot()
    
    stations = []
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
    
    return stations

def filter_stations_by_distance(stations, max_distance):
    return [station for station in stations if station['distance'] <= max_distance]


def load_stations(config, event_latitude, event_longitude, configured_stations):
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
        print(f"Loaded {len(stations)} configured stations within {max_distance} degrees from FDSN web service.")
        if stations:
            print(f"Station distances range from {min([s['distance'] for s in stations]):.2f} to {max([s['distance'] for s in stations]):.2f} degrees.")
        else:
            print("Warning: No stations were loaded. Check your station inventory and FDSN web service.")
        return sorted(stations, key=lambda x: x['distance'])
    except requests.RequestException as e:
        print(f"Error fetching data from FDSN web service: {e}")
        return []
    
def load_config(config_file):
    config = configparser.ConfigParser()
    config.optionxform = str  # This preserves the case of the keys
    config.read(config_file)
    return config

def generate_event_id(agency_id_lowercase, event_time):
    current_year = event_time.year
    year_start = datetime.datetime(current_year, 1, 1)
    year_fraction = (event_time - year_start).total_seconds() / (366 * 24 * 60 * 60)
    letter_value = int(year_fraction * (26**6))
    letters = ''.join(string.ascii_lowercase[(letter_value // (26**i)) % 26] for i in range(5, -1, -1))
    return f"{agency_id_lowercase}{current_year}{letters}"

def seiscomp_time_to_datetime(sc_time):
    return datetime.datetime.strptime(sc_time.toString("%Y-%m-%d %H:%M:%S.%f"), "%Y-%m-%d %H:%M:%S.%f")


def create_pick(origin, station, distance_km, phase, config, depth_km):
    pick = datamodel.Pick.Create()
    model = TauPyModel(model="iasp91")
    
    try:
        arrivals = model.get_travel_times(source_depth_in_km=depth_km,
                                          distance_in_degree=station['distance'],
                                          phase_list=[phase])
    except Exception as e:
        print(f"Error in TauPyModel for station {station['code']}, phase {phase}: {str(e)}")
        return None
    
    if not arrivals:
        print(f"No arrivals found for station {station['code']}, phase {phase}")
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

def create_arrival(pick, distance_deg, azimuth, phase, config, theoretical_time):
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

def create_picks_and_arrivals(origin, stations, config):
    picks = []
    arrivals = []
    used_stations = []
    origin_lat, origin_lon = origin.latitude().value(), origin.longitude().value()
    s_wave_cutoff = float(config['Phases']['s_wave_cutoff'])
    model = TauPyModel(model="iasp91")
    
    # Convert depth from meters to kilometers
    depth_km = origin.depth().value() / 1000
    print(f"Debug: Origin depth = {depth_km:.2f} km")
    
    max_distance = float(config['QualityParameters']['maximum_distance'])
    min_distance = float(config['Inventory']['min_distance'])
    
    print(f"Processing {len(stations)} stations")
    for idx, station in enumerate(stations):
        if idx % 10 == 0:
            print(f"Processing station {idx+1}/{len(stations)}")
        distance_deg = station['distance']
        azimuth = station['azimuth']
        
        if not (min_distance <= distance_deg <= max_distance):
            continue
        
        if station['code'] in used_stations:
            continue
        
        used_stations.append(station['code'])
        phases_to_pick = ['P', 'S'] if distance_deg <= s_wave_cutoff else ['P']
        
        for phase in phases_to_pick:
            print(f"Debug: Creating pick for station {station['code']}, phase {phase}, origin depth = {depth_km:.2f} km")
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
    
    print(f"Created {len(picks)} picks and {len(arrivals)} arrivals for {len(used_stations)} stations")
    return picks, arrivals, used_stations


def create_station_magnitudes(origin, magnitude, stations, config, used_stations):
    station_magnitudes = []
    used_stations_set = set(used_stations)
    print(f"Creating station magnitudes for magnitude type {magnitude.type()}")
    
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
        
        print(f"Created station magnitude {sta_mag.magnitude().value():.2f} for station {station['network']}.{station['code']}")
        
        # Remove the station from the set to ensure it's not used again
        used_stations_set.remove(station['code'])
        
        # Break if we've used all available stations
        if not used_stations_set:
            break
    
    print(f"Created {len(station_magnitudes)} station magnitudes")
    return station_magnitudes

def create_multiple_origins(config, event_time, stations, event_depth_km):
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

    print(f"Creating {num_origins} origins")
    print(f"Total available stations: {len(stations)}")
    print(f"Stations within distance range {min_distance}-{max_distance}: {len([s for s in stations if min_distance <= s['distance'] <= max_distance])}")

    if not stations:
        print("Warning: No stations available. Check your station inventory and FDSN web service.")
        return [], [], []

    for i in range(num_origins):
        print(f"\nCreating origin {i+1}/{num_origins}")
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
        origin.setDepth(datamodel.RealQuantity(depth_value, depth_uncertainty))  # Keep in km
        
        origin.setTime(datamodel.TimeQuantity(event_time))
        creation_time = base_creation_time + core.TimeSpan(i * creation_time_increment)
        creation_info = datamodel.CreationInfo()
        creation_info.setCreationTime(creation_time)
        creation_info.setAgencyID(config['Agency']['id'])
        creation_info.setAuthor(f"OriginLocator@{config['Agency']['id']}")
        origin.setCreationInfo(creation_info)
        origin.setEvaluationMode(datamodel.AUTOMATIC)
        origin.setEvaluationStatus(datamodel.PRELIMINARY)
        
        print(f"Origin depth: {depth_value:.2f} km")
        
        # Calculate the number of stations to use for this origin
        num_stations = min(initial_station_count + i * station_increase_per_origin, len(stations))
        
        # Select stations, prioritizing closer ones but also including some distant ones
        close_station_ratio = 0.7  # 70% of stations will be the closest ones
        close_station_count = int(num_stations * close_station_ratio)
        distant_station_count = num_stations - close_station_count

        close_stations = sorted_stations[:close_station_count]
        distant_stations = sorted_stations[close_station_count:]
        
        # Randomly select distant stations
        if distant_station_count > 0:
            distant_stations = random.sample(distant_stations, min(distant_station_count, len(distant_stations)))
        
        stations_to_use = close_stations + distant_stations
        
        # Filter stations by distance range
        stations_to_use = [s for s in stations_to_use if min_distance <= s['distance'] <= max_distance]
        
        print(f"Stations available for this origin: {len(stations_to_use)}")
        if stations_to_use:
            print(f"Distance range of used stations: {min([s['distance'] for s in stations_to_use]):.2f} to {max([s['distance'] for s in stations_to_use]):.2f} degrees")
        
        if not stations_to_use:
            print(f"Warning: No stations available for origin {i+1}. Skipping this origin.")
            continue
        
        picks, arrivals, used_stations = create_picks_and_arrivals(origin, stations_to_use, config)
        print(f"Created {len(picks)} picks and {len(arrivals)} arrivals")
        all_picks.extend(picks)
        for arrival in arrivals:
            origin.add(arrival)
        
        print("Setting origin quality")
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
            print("Warning: No stations available for setting distance parameters in origin quality.")
        
        origin.setQuality(quality)
        
        origins.append(origin)
        all_used_stations.append(used_stations)
        print(f"Finished creating origin {i+1}/{num_origins} using {len(stations_to_use)} stations")

    print(f"\nCreated {len(origins)} origins with {len(all_picks)} total picks")
    return origins, all_picks, all_used_stations

# Make sure to update the create_magnitudes and create_station_magnitudes functions if necessary
def create_magnitudes(origin, stations, config, focal_mechanisms, used_stations):
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


def create_mww_magnitudes(origins, config):
    mww_magnitudes = []
    centroid_origins = []
    mww_values = [float(val.strip()) for val in config['Magnitudes']['Mww'].split(',')]
    
    if len(mww_values) != 3:
        print("Warning: Expected 3 Mww values in config. Using default values.")
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
        centroid_depth = origin.depth().value() + random.uniform(-5, 5)
        centroid_time = origin.time().value() + core.TimeSpan(random.uniform(-5, 5))
        
        centroid.setLatitude(datamodel.RealQuantity(centroid_lat, 0.05))
        centroid.setLongitude(datamodel.RealQuantity(centroid_lon, 0.05))
        centroid.setDepth(datamodel.RealQuantity(centroid_depth, 2.5))
        centroid.setTime(datamodel.TimeQuantity(centroid_time))
        
        # Set centroid as the derived origin for the Mww magnitude
        mww_mag.setDerivedOriginID(centroid.publicID())
        
        # Add centroid information as a comment to Mww magnitude
        centroid_comment = datamodel.Comment()
        centroid_comment.setText(f"Centroid: Lat={centroid_lat:.4f}, Lon={centroid_lon:.4f}, Depth={centroid_depth:.2f}, Time={centroid_time.toString('%Y-%m-%d %H:%M:%S.%f')}")
        mww_mag.add(centroid_comment)
        
        mww_magnitudes.append(mww_mag)
        centroid_origins.append(centroid)

    return mww_magnitudes, centroid_origins

def create_focal_mechanisms(config, origins, event_id):
    focal_mechanisms = []
    mww_magnitudes = []
    centroid_origins = []
    fm_count = int(config['FocalMechanism'].get('count', 1))
    agency_id = config['Agency']['id']
    
    mww_values = [float(val.strip()) for val in config['Magnitudes']['Mww'].split(',')]
    if len(mww_values) != fm_count:
        print(f"Warning: Expected {fm_count} Mww values in config. Using default values.")
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

def create_moment_tensor(config, origin, event_id, index, mw):
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
    centroid.setDepth(datamodel.RealQuantity(centroid_depth_km, 2.5))
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

def calculate_tensor_components(scalar_moment, strike, dip, rake):
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

def calculate_azimuthal_gap(stations, used_stations):
    if not used_stations:
        return 360.0  # Full gap if no stations
    
    station_dict = {s['code']: s['azimuth'] for s in stations}
    azimuths = sorted([station_dict[code] for code in used_stations if code in station_dict])
    
    if not azimuths:
        return 360.0  # Full gap if no valid azimuths
    
    gaps = np.diff(azimuths)
    gaps = np.append(gaps, 360 + azimuths[0] - azimuths[-1])
    return np.max(gaps)

def create_synthetic_event(config, seiscomp_config_file):
    ep = datamodel.EventParameters()

    agency_id = config['Agency']['id']
    agency_id_lowercase = config['Agency']['id_lowercase']

    if config['Event']['time'].lower() == 'now':
        event_time = core.Time.GMT()
    else:
        event_time = core.Time.FromString(config['Event']['time'])

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

    configured_stations = parse_config_xml(seiscomp_config_file)
    global_stations = load_stations(config, event_latitude, event_longitude, configured_stations)
    print(f"Loaded {len(global_stations)} configured stations for event generation")

    print("Creating multiple origins")
    origins, picks, all_used_stations = create_multiple_origins(config, event_time, global_stations, event_depth_km)
    print(f"Created {len(origins)} origins and {len(picks)} picks")

    if not origins:
        print("Warning: No origins were created. Cannot proceed with event generation.")
        return ep

    print("Creating focal mechanisms")
    focal_mechanisms, mww_magnitudes, centroid_origins = create_focal_mechanisms(config, origins, event_id)
    print(f"Created {len(focal_mechanisms)} focal mechanisms with Mww magnitudes and centroids")
    
    print("Adding picks to event parameters")
    for pick in picks:
        ep.add(pick)

    print("Adding origins, magnitudes, and arrivals to event parameters")
    for origin, used_stations in zip(origins, all_used_stations):
        ep.add(origin)
        event.add(datamodel.OriginReference(origin.publicID()))

        print(f"Creating magnitudes for origin {origin.publicID()}")
        mags = create_magnitudes(origin, global_stations, config, focal_mechanisms, used_stations)
        for mag in mags:
            origin.add(mag)
            print(f"Created magnitude: {mag.type()} = {mag.magnitude().value():.2f}")

        print(f"Creating station magnitudes for origin {origin.publicID()}")
        used_stations_copy = used_stations.copy()
        for mag in mags:
            station_magnitudes = create_station_magnitudes(origin, mag, global_stations, config, used_stations_copy)
            for sta_mag in station_magnitudes:
                origin.add(sta_mag)
        print(f"Created {len(station_magnitudes)} station magnitudes for origin {origin.publicID()}")

    if focal_mechanisms:
        print("Adding focal mechanisms, Mww magnitudes, and centroids to event parameters")
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
            
            print(f"Added focal mechanism {fm.publicID()} with Mww {mww_mag.magnitude().value():.2f}")
            print(f"  Centroid: Lat={centroid.latitude().value():.4f}, Lon={centroid.longitude().value():.4f}, Depth={centroid.depth().value():.2f} km")

    # Set preferred entities
    if origins:
        preferred_origin = origins[-1]
        event.setPreferredOriginID(preferred_origin.publicID())
    
    if mww_magnitudes:
        event.setPreferredMagnitudeID(mww_magnitudes[-1].publicID())
    
    if focal_mechanisms:
        event.setPreferredFocalMechanismID(focal_mechanisms[-1].publicID())


    print(f"\nCreated event with ID: {event.publicID()}")
    print(f"Preferred Origin ID: {event.preferredOriginID()}")
    print(f"Preferred Magnitude ID: {event.preferredMagnitudeID()}")
    print(f"Preferred Focal Mechanism ID: {event.preferredFocalMechanismID()}")

    # Print summary of the preferred origin
    preferred_origin = ep.findOrigin(event.preferredOriginID())
    if preferred_origin:
        print("\nPreferred Origin Summary:")
        print(f"  Time: {preferred_origin.time().value().toString('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"  Latitude: {preferred_origin.latitude().value():.4f} ± {preferred_origin.latitude().uncertainty():.4f}")
        print(f"  Longitude: {preferred_origin.longitude().value():.4f} ± {preferred_origin.longitude().uncertainty():.4f}")
        print(f"  Depth: {preferred_origin.depth().value():.2f} ± {preferred_origin.depth().uncertainty():.2f} km")
        print(f"  Evaluation Mode: {preferred_origin.evaluationMode()}")
        print(f"  Evaluation Status: {preferred_origin.evaluationStatus()}")
        
        print("\n  Magnitudes:")
        for i in range(preferred_origin.magnitudeCount()):
            mag = preferred_origin.magnitude(i)
            print(f"    {mag.type()}: {mag.magnitude().value():.2f} ± {mag.magnitude().uncertainty():.2f}")
            if mag.type() == "Mww":
                for j in range(mag.commentCount()):
                    comment = mag.comment(j)
                    if comment.text().startswith("Mww solution"):
                        print(f"      {comment.text()}")

    # Print summary of the preferred focal mechanism
    preferred_fm = ep.findFocalMechanism(event.preferredFocalMechanismID())
    if preferred_fm:
        print("\nPreferred Focal Mechanism Summary:")
        print(f"  ID: {preferred_fm.publicID()}")
        np = preferred_fm.nodalPlanes()
        if np:
            np1 = np.nodalPlane1()
            print(f"  Nodal Plane 1: Strike {np1.strike().value():.1f}° ± {np1.strike().uncertainty():.1f}°, "
                  f"Dip {np1.dip().value():.1f}° ± {np1.dip().uncertainty():.1f}°, "
                  f"Rake {np1.rake().value():.1f}° ± {np1.rake().uncertainty():.1f}°")
        
        if preferred_fm.momentTensorCount() > 0:
            mt = preferred_fm.momentTensor(0)
            print("\n  Moment Tensor:")
            print(f"    Derived Origin ID: {mt.derivedOriginID()}")
            tensor = mt.tensor()
            print(f"    Scalar Moment: {mt.scalarMoment().value():.2e} Nm")
            print(f"    Double Couple: {mt.doubleCouple():.2f}")
            print(f"    CLVD: {mt.clvd():.2f}")
            print(f"    Moment Magnitude ID: {mt.momentMagnitudeID()}")
        
        # Find the associated Mww magnitude
        centroid_origin = ep.findOrigin(mt.derivedOriginID())
        if centroid_origin:
            for i in range(centroid_origin.magnitudeCount()):
                mag = centroid_origin.magnitude(i)
                if mag.type() == "Mww":
                    print(f"\n  Associated Mww Magnitude:")
                    print(f"    Value: {mag.magnitude().value():.2f} ± {mag.magnitude().uncertainty():.2f}")
                    print(f"    Method: {mag.methodID()}")
                    break
            else:
                print("\n  No associated Mww magnitude found")
        else:
            print("\n  No associated centroid origin found")

    return ep

def write_to_xml(ep, filename_prefix):
    # Get the event ID from the EventParameters object
    event_id = ep.event(0).publicID() if ep.eventCount() > 0 else "unknown"
    
    # Create a filename with the event ID
    filename = f"{filename_prefix}_{event_id}.xml"
    
    ar = io.XMLArchive()
    if not ar.create(filename):
        print(f"Could not create file: {filename}")
        return False
    ar.setFormattedOutput(True)
    ar.writeObject(ep)
    ar.close()
    print(f"Event has been written to {filename}")
    return True


def main():
    config = load_config('config.ini')

    try:
        ep = create_synthetic_event(config, 'config.xml')
        print("Successfully created synthetic event")

        # Use a prefix for the output file
        output_file_prefix = "synthetic_event_seiscomp"
        if write_to_xml(ep, output_file_prefix):
            print("Event has been written to file")
        else:
            print("Failed to write event to file")

        # Print summary of generated event
        for i in range(ep.eventCount()):
            event = ep.event(i)
            print(f"\nEvent {i+1} Summary:")
            print(f"ID: {event.publicID()}")
            print(f"Type: {event.type()}")
            print(f"Creation Time: {event.creationInfo().creationTime().toString('%Y-%m-%d %H:%M:%S.%f')}")
            
            preferred_origin = ep.findOrigin(event.preferredOriginID())
            if preferred_origin:
                print("\nPreferred Origin:")
                print(f"  ID: {preferred_origin.publicID()}")
                print(f"  Time: {preferred_origin.time().value().toString('%Y-%m-%d %H:%M:%S.%f')}")
                print(f"  Latitude: {preferred_origin.latitude().value():.4f} ± {preferred_origin.latitude().uncertainty():.4f}")
                print(f"  Longitude: {preferred_origin.longitude().value():.4f} ± {preferred_origin.longitude().uncertainty():.4f}")
                print(f"  Depth: {preferred_origin.depth().value():.2f} ± {preferred_origin.depth().uncertainty():.2f} km")
                print(f"  Evaluation Mode: {preferred_origin.evaluationMode()}")
                print(f"  Evaluation Status: {preferred_origin.evaluationStatus()}")
                
                print("\n  Magnitudes:")
                for j in range(preferred_origin.magnitudeCount()):
                    magnitude = preferred_origin.magnitude(j)
                    if magnitude.magnitude().uncertainty() is not None:
                        print(f"    {magnitude.type()}: {magnitude.magnitude().value():.2f} ± {magnitude.magnitude().uncertainty():.2f}")
                    else:
                        print(f"    {magnitude.type()}: {magnitude.magnitude().value():.2f} (uncertainty not set)")
            
            preferred_fm = ep.findFocalMechanism(event.preferredFocalMechanismID())
            if preferred_fm:
                print("\nPreferred Focal Mechanism:")
                print(f"  ID: {preferred_fm.publicID()}")
                print(f"  Triggering Origin ID: {preferred_fm.triggeringOriginID()}")
                np = preferred_fm.nodalPlanes()
                if np:
                    np1 = np.nodalPlane1()
                    print(f"  Nodal Plane 1: Strike {np1.strike().value():.1f}° ± {np1.strike().uncertainty():.1f}°, "
                          f"Dip {np1.dip().value():.1f}° ± {np1.dip().uncertainty():.1f}°, "
                          f"Rake {np1.rake().value():.1f}° ± {np1.rake().uncertainty():.1f}°")
                
                if preferred_fm.momentTensorCount() > 0:
                    mt = preferred_fm.momentTensor(0)
                    print("\n  Moment Tensor:")
                    print(f"    Derived Origin ID: {mt.derivedOriginID()}")
                    tensor = mt.tensor()
                    print(f"    Scalar Moment: {mt.scalarMoment().value():.2e} Nm")
                    print(f"    Double Couple: {mt.doubleCouple():.2f}")
                    print(f"    CLVD: {mt.clvd():.2f}")
                    print(f"    Moment Magnitude ID: {mt.momentMagnitudeID()}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()