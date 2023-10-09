# Dataset Description
The scenario used to generate this dataset is the one provided in the following paper:

C. Obermaier, R. Riebl and C. Facchi, "Dynamic scenario control for VANET simulations," 2017 5th IEEE International Conference on Models and Technologies for Intelligent Transportation Systems (MT-ITS), Naples, Italy, 2017, pp. 681-686, doi: 10.1109/MTITS.2017.8005599.

| COLUMN NAME   | MEANING |
|   ---         |   ---   |
| source        | stationID identifier of the ITS-S that generates the ITS message in question.|
| destination   | destination of the ITS message (null when broadcast).|
| messageID     |   Type of the ITS message [1]. <br> - (1) **denm**: Decentralized Environmental Notification Message (DENM) [2]; <br> - (2) **cam**: Cooperative Awareness Message (CAM) [3]; <br> - (3) **poi**: Point of Interest Message [4]; <br> - (4) **spat**: Signal Phase And Timing (SPAT) Message [5]; <br> - (5) **map**: MAP Message [5]; <br> - (6) **vi**: In Vehicle Information (IVI) Message [6]; <br> - (7) **ev-rsr**: Electric vehicle recharging spot reservation Message.|
| actionID      | Event identification. <br> ``stationID:seqNumber`` <br> Sequence number assigned to the *actionID* for each new event. <br> **N.B.** In case of multiple originating ITS-S detect the same event for the first time, the *actionID* should be different in each originating ITS-S.|
|situation_informationQ| Situation information quality.  It indicates the probability of the detected event being truly existent at the event position.|
|situation_eventType| Description for the event type, including direct cause and sub cause. **In this case only direct cause is provided.** See [Cause Code section](#cause-code).|
|location_speed\*       | Moving speed of a detected event and the confidence of the moving speed information.|
|location_heading\*      | The heading direction of the event and the confidence of the heading information, if applicable.
|detection_time\*\*          | Time at which the event is detected by the originating ITS-S.<br> For the DENM repetition, this DE shall remain unchanged.<br> For the DENM update, this DE shall be the time at which the event update is detected. <br> For the DENM termination, this DE shall be the time at which the termination of the event is detected. |
|reference_time             | The referenceTime represents the time at which a DENM is generated by the DEN basic service, after receiving the application request. |
|simulation_time (secs)            | It represents the simulation time at which the DENM message is received by a ITS-s. Start is 0.|
|eventPos_lat               | Geographical position of the detected event (Latitude). |
|eventPos_long               | Geographical position of the detected event (Longitude). |
|eventPos_alt               | Geographical position of the detected event (Altitude). |
|relevanceDistance          | The distance in which event information is relevant for the receiving ITS-S. **Not set in this scenario**|
|termination				| It can assume two values: <br> ``` Termination_isCancellation	= 0, Termination_isNegation	= 1 ``` <br> - **Cancellation**: A DENM that informs the termination of an event. A cancellation DENM is transmitted by the same originating ITS-S which has generated the new DENM for the same event. <br> - **Negation**: A DENM that informs the termination of an event for which the new DENM has been received by the originating ITS-S from another ITS-S. A negation DENM may be used to announce thetermination of an event if the originating ITS-S has the capacity to detect the termination of an event which has been previously announced by other ITS-Ss| 
|stationType\*\*\*                | Provides the station type information of the originating ITS-S.|






<sub>\* In this case only the value is reported (no confidence information) </sub>

<sub>\*\* The parameter referenceTime is the identifier for DENM update referring to a specific actionID. The *referenceTime* represents the time at which a DENM is generated by the DEN basic service, after receiving the application request. For each DENM update, the *referenceTime* shall be updated and the value shall be greater than the *referenceTime* value of
the previous DENM update for the same actionID. The actionID shall remain unchanged for DENM update, as long as the stationID of the originating ITS-S remains
unchanged.</sub>

<sub>\*\*\* In the dataset the stationType is 0, however, the type of the station is vehicle.</sub>
<br>
<br>

**N.B.** 
* The dataset provided considers only DENM messages.
* ITS-s means ITS station and includes any kind of station type defined in the [following section](#stationtype).


## References
---
[1] https://www.etsi.org/deliver/etsi_ts/102800_102899/10289402/01.02.01_60/ts_10289402v010201p.pdf

[2] https://www.etsi.org/deliver/etsi_en/302600_302699/30263703/01.03.01_60/en_30263703v010301p.pdf

[3] https://www.etsi.org/deliver/etsi_en/302600_302699/30263702/01.04.01_60/en_30263702v010401p.pdf

[4] https://www.etsi.org/deliver/etsi_ts/101500_101599/10155601/01.01.01_60/ts_10155601v010101p.pdf

[5] https://transportationops.org/sites/transops/files/SPaT%20V2I%20Interface%20Requirements%2020200409_circulate.pdf

[6] https://www.iso.org/obp/ui/#iso:std:iso:ts:19321:ed-2:v1:en

[7] https://www.etsi.org/deliver/etsi_ts/101500_101599/10155603/01.01.01_60/ts_10155603v010101p.pdf


### Other sources used
Content of this README file comes from the specification of the DENM message (reference [1]).

---
## Cause Code
```
CauseCodeType_reserved	= 0,

CauseCodeType_trafficCondition	= 1,

CauseCodeType_accident	= 2,

CauseCodeType_roadworks	= 3,

CauseCodeType_impassability	= 5,

CauseCodeType_adverseWeatherCondition_Adhesion	= 6,

CauseCodeType_aquaplannning	= 7,

CauseCodeType_hazardousLocation_SurfaceCondition	= 9,

CauseCodeType_hazardousLocation_ObstacleOnTheRoad	= 10,

CauseCodeType_hazardousLocation_AnimalOnTheRoad	= 11,

CauseCodeType_humanPresenceOnTheRoad	= 12,

CauseCodeType_wrongWayDriving	= 14,

CauseCodeType_rescueAndRecoveryWorkInProgress	= 15,

CauseCodeType_adverseWeatherCondition_ExtremeWeatherCondition	= 17,

CauseCodeType_adverseWeatherCondition_Visibility	= 18,

CauseCodeType_adverseWeatherCondition_Precipitation	= 19,

CauseCodeType_slowVehicle	= 26,

CauseCodeType_dangerousEndOfQueue	= 27,

CauseCodeType_vehicleBreakdown	= 91,

CauseCodeType_postCrash	= 92,

CauseCodeType_humanProblem	= 93,

CauseCodeType_stationaryVehicle	= 94,

CauseCodeType_emergencyVehicleApproaching	= 95,

CauseCodeType_hazardousLocation_DangerousCurve	= 96,

CauseCodeType_collisionRisk	= 97,

CauseCodeType_signalViolation	= 98,

CauseCodeType_dangerousSituation	= 99
```

## StationType
```
	StationType_unknown	= 0,
	StationType_pedestrian	= 1,
	StationType_cyclist	= 2,
	StationType_moped	= 3,
	StationType_motorcycle	= 4,
	StationType_passengerCar	= 5,
	StationType_bus	= 6,
	StationType_lightTruck	= 7,
	StationType_heavyTruck	= 8,
	StationType_trailer	= 9,
	StationType_specialVehicles	= 10,
	StationType_tram	= 11,
	StationType_roadSideUnit	= 15
```

