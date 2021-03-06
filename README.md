# Rapid Comfort Testing (RCT)
▶️ [Download thesis](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2828748) 
| ✔️ Version: 1.0 
| Submitted by: TK

🛠️ Faculty of Engineering | 📚 Department of Civil and Environmental Engineering

The goal of this project is to explore the potential for applying artificial neural networks (ANNs) to
predict both annual daylight illuminance and operative temperature, in order to reduce time-consuming
simulation methods. Promising machine learning approaches from the literature are implemented and
evaluated for performance.

## SCOPE OF THE STUDY
This study investigates the potential for applying ANNs to predict both annual daylight illuminance
and operative temperature in residential buildings. In current practice, both daylight illuminance and
operative temperature are treated separately, which leads to increased time and costs for new building
projects. Tools that address the problem are in short supply and the task is motivated by solving this
challenge.
This study aims to establish a basis for a learning algorithm to partially replace time-consuming BPS
tools in the design optimization processes. To the best of the authors’ knowledge, there are no studies
that use ANNs to predict annual climate-based metrics, for both daylight illuminance and operative
temperature, in parametric design environments. This thesis aims to introduce a proof of concept,
which predicts annual hourly illuminance and operative temperature for a grid of sensor-nodes in a
zone, based on geometric data. The originality of this research lies within the following objectives:
1. Review state-of-the-art research, related to ANNs, thermal and visual comfort.
2. Deploy a BPS model with grid-based calculations, for annual daylight illuminance and
operative temperature.
3. Develop a machine learning algorithm, which can partially replace and reduce the timeconsuming simulations methods, in order to achieve multi-objective targets.
The methodology consists of generating training, validation and testing data by using parametric
analysis and exhaustive search. The generated data is based on theoretical reference building and
represent middle-class housing of the existing buildings in Norway. A custom ANN will be developed
with suitable model architectures and hyper-parameters. Validating and testing the ANN results will
also be addressed.

![image](https://user-images.githubusercontent.com/79943730/155010020-959a5215-2020-42e5-a66e-da8ac01a9bd9.png)

## DETAILS
Humans spend 87 % of their time indoors, mostly in their own residence. The indoor environment is a
crucial factor for people’s health and welfare. There is an increasing challenge with overheating in
buildings due to hotter climate. In addition, centralization and stricter building codes has led us to
build more compact, making it more challenging to achieve daylight criteria in new building projects.
Building Performance Simulation (BPS) is proving to be an effective approach for supporting the
design and finding a balance between daylight availability, thermal comfort and energy performance.
In current practice these aspects are treated separately, which leads to increased time and costs in
building projects. Tools that address the problem are in short supply and the task is motivated by
solving this challenge. The use of artificial neural networks (ANNs) promises great support and
improved feasibility to BPS, due to a reduction in overall computation time.

This project investigates the potential for applying ANNs to predict both annual daylight illuminance
and operative temperature. The main findings from deploying a simulation model is the importance of
multi sensor-node calculations for operative temperature. Operative temperature is usually calculated
for the room center, in contrast to daylight where illuminance is calculated for a grid of sensor-nodes.
In this study, operative temperature including long- and shortwave radiation have been calculated for a
grid of sensor-nodes. The results show a significant difference in operative temperature at different
locations in the room where shortwave radiation has greatest impact on the results. It is therefore
important to address operative temperature in the same way as daylight illuminance, using a grid of
sensor-nodes when exploring multi-objective optimization performance. However, these calculations
are computational demanding and increase simulation time by 2000 %. The author has therefore
investigated the potential for applying machine learning techniques, to partially replace and reduce the
time-consuming simulations methods in order to achieve multi-objective design targets.

A fully connected neural network is developed with five hidden layers and five different neuron
structures. The ANN model for operative temperature performed overall best for predicting annual
values, reaching a CV(RMSE) of 3.8 %, an accuracy of 98 % and an average prediction within
0.47 °C. The ANN model for daylight is less accurate. The results show that direct sun exposure is
difficult to predict with a five-layer ANN structure and the model often underestimates these
variations. The overall model is precise but not accurate, meaning it is following the same pattern, but
is consequently predicting lower temperature and illuminance values.
In general, the ANN models are showing promising results which may be integrated in a multiobjective design workflow. The results show significant time saving potential by implementing ANNs.
The overall time is reduced by 96 % by using ANN models for predicting annual temperature and
illuminance values
