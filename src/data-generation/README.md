# data-generation
✔️ Version: 1.0 
| 📚 Category: Grasshopper
| 🛠️ Discipline: RIEn
| Submitted by: TK

## 🧰 Description 
The first step of the methodology, numbered “1a” in Figure 5, is to develop the BPS model to generate 
training, validation and test data. It would be preferable to use real measured data from existing 
buildings for training and validation, but this would be cost- and time-consuming. A building model is
therefore developed using BPS tools, which is validated with real building models. Explorative use of 
BPS tools, for research and development, might be a potential pitfall (Loonen et al., 2019). Although 
these tools are mainly used for generating generic room models and not extreme model cases.
Figure 5 Flow diagram of the process presented in this thesis.

![image](https://user-images.githubusercontent.com/81426268/155008544-c700436a-7a30-43ee-9289-1ac2f8aed458.png)


## 👋 Quickstart
- Rhinoceros version 6 SR34
- Grasshopper graphical algorithm editor version 1.0
- [Ladybug tools 1.2.0](https://www.food4rhino.com/en/app/ladybug-tools)
- [TT Toolbox 1.9](https://www.food4rhino.com/en/app/tt-toolbox)

## ⚙️ Setup
The generic simulation model is based on theoretical reference building and represent a room form a
middle-class housing in Norway. The basic geometric properties are based on test-cases from the CIE 
171:2006 standard and the ANSI/ASHRAE Standard 140-2001 (CIE, 2006, ASHRAE, 2001). The 
model is slightly modified, in order to represent a combination for daylight illuminance and operative 
temperature. This is done for the purpose of comparison between simulated and predicted results. The 
generic model consists of one rectangular room (3.0 m wide, 6.0 m long and 2.7 m high), with no 
interior partitions and one-sided window, se illustration in Figure 6. 

Building geometry and parametric variables are designed in Grasshopper graphical algorithm editor
version 1.0. The editor is a built-in plugin in Rhinoceros 3D modeling tool, which is widely used by 
architects and engineers, and is interoperable with BIM tools. Rhinoceros version 6 SR34 is used.
Ladybug tools version 1.2.0, which is integrated into Grasshopper, is used for BPS.
Two scripts are developed for generating training data, one for visual comfort and calculation daylight
illuminance, and another for thermal comfort and calculation operative temperature. The graphical
color-coding standard (Dynamo Standard) for Grasshopper, developed by Vladimir Ondejcik, is
implemented as a part of the code validation and readability (Ondejcik, 2016). An overview of the 
scrips, developed in Grasshopper, can be found in Appendix 2.

Operative temperature is a common thermal comfort metric and is chosen as output result for this 
study. This is because it is connected to the Norwegian building standard and used for several other 
common thermal comfort metrices, such as PMV, TCV and TSV, as described in the theory section. 
Operative temperature is calculated based on dry bulb air temperature, mean radiant temperature and 
air velocity.

Annual daylight illuminance is a common visual comfort metric and is chosen as output result for this 
study. This is because it is connected to several common daylight metrices such as UDI and DA and 
provides the model with a more flexible usage. Illuminance is the measure of the amount of light 
received on a surface, and is typically expressed in lux (lm/m2
) (Andersen et al., 2014).

![image](https://user-images.githubusercontent.com/81426268/155008751-625f21e3-1749-4939-9e8d-aa2d02f6bd45.png)

It is found from the literature study that most ANNs are trained to predict a single output value. This 
can be metrices such as DA and PMV. In this study the goal is to utilize ANNs and use annual hourly 
data for several sensor nodes, representing every position in the room. This is because in buildings where 
there is significant solar radiation, air temperatures and radiant temperatures may be very different 
(Myhren and Holmberg, 2006). The operative temperature is therefore calculated as a grid of sensor-nodes in the same way as daylight illuminance. This makes it possible to take account for radiant surface temperatures (longwave radiation) at different positions in the zone. In addition, the effect of direct sun 
exposure (shortwave radiation) is included for the operative temperature. This is done by calculating 
mean radiant temperature (MRT) for each position using the SolarCal model of ASHRAE-55 (ASHRAE, 2017). This method estimates the effects of shortwave solar and sky exposure to determine longwave radiant exchange. It is assumed that the whole body is irradiated if the sensor-node is 
irradiated. 

Sensor-nodes have been set with 0.5 m distance between nodes, and 0.5 m from walls based on NS-EN 12464-1:2011 (Norsk Standard, 2011), resulting in a total of 40 sensor-nodes for each model. Figure 7 illustrates the location of the sensor-nodes. The Norwegian building research design guides
from SINTEF suggest a distance of 0.6 m from walls for thermal models (Mysen, 2017). This is
reduced to 0.5 m, in order to compare the results with annual daylight illuminance. There is no 
furniture present in the room.
