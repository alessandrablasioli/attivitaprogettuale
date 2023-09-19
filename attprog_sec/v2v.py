import argparse
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim
from datetime import datetime
import numpy as np
# import shapely
from shapely.geometry import shape, Point, MultiPoint
from shapely import centroid

from geopy import distance

# The following dictionary contains the time and radius thresholds used to 
# understand if the DEN message received is too old or too far. This depends on the eventType
# eventype: threshold (secs,meters)
thresholds = {
    1: (2,4), # trafficCondition
    2: 3600, # accident
    3: None, # roadworks
    5: None, # impassability
    6: 86400, # adverseWeatherCondition_Adhesion
    7: 86400, # aquaplannning
    9: 3600, # hazardousLocation_SurfaceCondition
    10: 3600, # hazardousLocation_ObstacleOnTheRoad
    11: 3600, # hazardousLocation_AnimalOnTheRoad
    12: 600, # humanPresenceOnTheRoad
    14: 600, # wrongWayDriving
    15: 3600, # rescueAndRecoveryWorkInProgress
    17: 3600, # adverseWeatherCondition_ExtremeWeatherCondition
    18: 1800, # adverseWeatherCondition_Visibility
    19: 1800, # adverseWeatherCondition_Precipitation
    26: 600, # slowVehicle
    27: 3600, # dangerousEndOfQueue
    91: 600, # vehicleBreakdown
    92: 3600, # postCrash
    93: 600, # humanProblem
    94: 600, # stationaryVehicle
    95: 120, # emergencyVehicleApproaching
    96: 30, # hazardousLocation_DangerousCurve
    97: 10, # collisionRisk
    98: 30, # signalViolation
    99: 30, # dangerousSituation
}

def calculate_space_centroid(points):
    length = len(points)
    sum_lon = np.sum(points[:, 0])
    sum_lat = np.sum(points[:, 1])
    return sum_lon/length, sum_lat/length



def check_distance(den_pos, cam_pos, r=50):
    return distance.distance(den_pos, cam_pos).m <= r

def check_cam(denm, cams):
    count = 0
    for cam in cams:
        d = distance.distance(denm, cam).m <= 100
        if not d:
            count += 1
            break
    print(count)

def check_cov_intersection(geojson, point):
    for feature in geojson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            print('Intersection found')
            return True
    
    return False

def find_similar_event(event_collector, event):
    eventType, p, t = event
    radius, time_threshold = thresholds[1]
    for key in event_collector:
        time_centroid = event_collector[key]["time_centroid"]
        space_centroid = event_collector[key]["space_centroid"]

        if event_collector[key]["eventType"] == eventType and check_distance(p, space_centroid) <= radius and abs(time_centroid - t) <= time_threshold:
            event_collector[key]["time_centroid"] = (event_collector[key]["time_centroid"] + t ) / 2
            event_collector[key]["space_centroid"] = centroid(MultiPoint([event_collector[key]["space_centroid"], p]))
            return key
    return None


def area_definition(point, radius):
    """
    The point represents the center of a circle
    These data are used to define an area to consider the reliability of a den message
    """
    print("Do something")


def compare_with_rsu(den):
    """
        This method returns True if the data received from that vehicle
        match with data provided by the RSU
    """
    return True

def main(args):
    
    # Loading parameters
    cam_time_window = args.time_window_cam * 1000
    geojson_file = args.area
    if not os.path.isfile(geojson_file):
        print("Coverage geojson file not found")
        exit()

    # Loading geojson file
    with open(geojson_file) as f:
        geojson = json.load(f)
    

    # Loading dataset
    denm = pd.read_csv("./DENM-dataset/datasetDen.csv", sep = ";")
    cam = pd.read_csv("./CAM-dataset/datasetCam.csv", sep = ";")
    denm['eventPos_long'] = denm['eventPos_long']/1e7
    denm['eventPos_lat'] = denm['eventPos_lat']/1e7

    cam['referencePositionLong'] = cam['referencePositionLong']/1e7
    cam['referencePositionLat'] = cam['referencePositionLat']/1e7

    # Data pre-processing
    # 1. SimTime starts from zero, we need it in ms TAI format
    tai_sync = datetime.strptime('2004-01-01 00:00:00', '%Y-%m-%d %H:%M:%S') 
    # The former value is usually based on your local timezone. We need to convert to UTC as the time used in the dataset
    utc_tai_sync = datetime.utcfromtimestamp(tai_sync.timestamp()) 
    temp_start_time = datetime.strptime('2017-06-26 12:00:00', '%Y-%m-%d %H:%M:%S')
    new_start_time = (temp_start_time.timestamp()*1000) - (utc_tai_sync.timestamp() * 1000) 
    denm['message_reception_time'] =  (denm['simulation_time']*1000) + new_start_time
    cam['message_reception_time'] =  (cam['simulationTime']*1000) + new_start_time

    # 2. Checking information quality
    denm = denm.loc[denm['situation_informationQ'] > 0.6]

    # 3. Group by simTime (receiving time), source, situationEventType
    # This excludes message propagations
    # In a simulated scenario we can apply drop duplicates as all the data in the columns are the same
    denm = denm.drop_duplicates()
    # TODO Should we also drop duplicates in CAM dataset?
    cam = cam.drop_duplicates()
    # In other scenarios we may also want to aggregate same messages generated from a certain source within a time interval

    # 4. TODO Cam preprocessing
    # deleting from the dataset messages not in the coverage area


    # 5. Order the dataset using time of message reception
    # We use this to simulate a real system
    denm = denm.sort_values(by=['message_reception_time'])
    cam = cam.sort_values(by=['message_reception_time'])

    event_collector = {}

    for index, row in denm.iterrows():

        # TODO Reputation-based selection
        # Exclude all messages sent from a source with low reputation score


        # Time-based selection
        # Exclude all the DEN messages with a detection time too old in terms of receiving time
        # this may depend on the type of event
        # We use the time when we receive the message as refernce point
        message_age = row['message_reception_time'] - row['detection_time']
        time_threshold = thresholds[row['situation_eventType']] * 1000
        if message_age > time_threshold:
            continue
        
        # Spatial-based selection
        # Ignore messages generated by vehicles not in the edge node area
        # This may represent two conditions:
        #   1. Problem on vehicle sensor, so better to avoid this data (decrease reputation (?))
        #   2. Disinformation attack
        p = Point(row['eventPos_long'], row['eventPos_lat'])


        if not check_cov_intersection(geojson, p):
            # TODO decide if we need to decrease the reputation score
            continue
        
        # Comparing DEN message information with information provided by RSU (if any)
        if not compare_with_rsu(row):
            # Here actually we should decrease the reputation 
            continue
        # For each DEN message received we need all the cam received in the past 'time_window' 
        # from the same source
        # TODO Should we consider the detection time or the reception time?
        cam_from_source = cam.loc[(cam['source']==row['source']) & (cam['message_reception_time'] < row['detection_time']) & (cam['message_reception_time'] >  row['detection_time'] - cam_time_window)]
        
        inherent_cam_counter = 0
        for index, cam_row in cam_from_source.iterrows():
            if check_distance(den_pos=(row['eventPos_lat'], row['eventPos_long']), cam_pos=(cam_row['referencePositionLat'], cam_row['referencePositionLong'])):
                inherent_cam_counter += 1

        coherency_percentage = (inherent_cam_counter/len(cam_from_source.index)) * 100
        print("Percentage of coherent cam messages {}%".format(coherency_percentage))
        eventType = row['situation_eventType']
        # area = area_definition(point=p, radius=5)
        t = row['detection_time']
        # Event type collector selection
        if event_collector:
            similar_event = find_similar_event(event_collector, (eventType,p,t))
            if similar_event:
                event_collector[similar_event]["dens"] += [(eventType, p, t)]
            else:
                event_collector[hash((eventType, p, t))] = {  "eventType" : eventType, 
                                                    "space_centroid" : p, 
                                                    "time_centroid" : t, 
                                                    "dens" : [(eventType, p, t)]}
        else:
            # (eventType, centroid, timestamp)
            event_collector[hash((eventType, p, t))] = {  "eventType" : eventType, 
                                                    "space_centroid" : p, 
                                                    "time_centroid" : t, 
                                                    "dens" : [(eventType, p, t)]}
        



        # if inherent_cam_counter > 0:
        #     print("{} out of {} inherent cooperative awareness messages found".format(inherent_cam_counter,len(cam_from_source.index)))

        
        


    



    # controlliamo che è vicino a me
    # controlliamo che sia nell'intorno del mio timpestamp
    # controlliamo l'information quality > IQ

    delta = 2 #minuti
    # aspettiamo un intervallo delta, e nel mentre continuiamo a ricevere messaggi
    N = 3
    # verifico che ci siano almeno N DENM (con source diversa)

    # NON CI SONO

    # controllo se esistono CAM nelle vicinanze (temporali e spaziali)

    # se ci sono:
    # se sono la maggioranza, allora quel DENM era falso

    # penalty

    # se non ci sono:
    # ignoro

    # CI SONO

    # li raggruppo rispetto all'evento a cui fanno riferimento
    # come capisco che sia lo stesso evento?

    # hanno stesso eventType, sono vicini temporalmente (dipende dal tipo) e spazialmente.
    # verifico che sia un evento vero in base ai CAM




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Disinformation attack detection algorithm for VANET alert messages")

    parser.add_argument('-wc', '--time_window_cam', metavar='<time in secs>',
                        help='time used for CAM sampling during DENM evaluation', type=float, default=600)
    
    parser.add_argument('-a', '--area', metavar='<geojson file path>',
                        help='geojson file defining the coverage of the edge node', type=str, default='coverage.json')
    
    

    args = parser.parse_args()

    main(args)

