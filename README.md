# Rapid Comfort Testing (RCT)
✔️ Version: 1.0 
| Submitted by: TK

🛠️ Faculty of Engineering | 📚 Department of Civil and Environmental Engineering

The goal of this project is to explore the potential for applying artificial neural networks (ANNs) to
predict both annual daylight illuminance and operative temperature, in order to reduce time-consuming
simulation methods. Promising machine learning approaches from the literature are implemented and
evaluated for performance.

## ABSTRACT 
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

## BACKGROUND
Humans spend 87 % of their time indoors, mostly in their own residence. The indoor environment is a
crucial factor for people’s health and welfare (Klepeis et al., 2001). It is well-known that windows
have a considerable impact on both energy use and indoor environment. Study shows that long-term
impact of attending a daylit school, could result in a 14 % increase in student performance in contrast
to a classroom without windows (Bailey and Nicklas, 1996). Buildings also account for 40 % of
energy use in the EU (EPBD, 2010). In recent years buildings in Norway have become more insulated
and airtight due to more ambitious building projects with certifications such as BREEAM and ZEB.
With hotter climate there has been increasingly challenge with overheating in buildings (Tian and
Hrynyszyn, 2020). In addition, centralization and stricter building codes has led us to build more
compact, making it more challenging to achieve daylight criteria in new building projects (Reinhart
and Selkowitz, 2006, Chen and Yang, 2015). 

Given the increasing complexity of energy and environmental challenges the building sector is facing,
Building Performance Simulation (BPS) is proving to be an effective approach for supporting the
design and operation of high-performance buildings, such as zero-energy buildings or zero-emission
buildings (Clarke and Hensen, 2015, Wate et al., 2019). It is important to find a balance between
daylight availability, thermal comfort and energy use, if we are to achieve both the goal of a nearly
zero energy consumption and buildings with a healthy and comfortable indoor environment (Yu et al.,
2020, Ruck et al., 2000).

Simulation-based multi-objective optimizations is widely applied when optimizing both thermal and
daylighting performance. Some of these methods are genetic algorithm, weighted sum method and
non-dominated sorting genetic algorithm II (NSGA-II). These algorithms offers a Pareto-optimal front
that shows the best trade-offs between daylight and thermal comfort objectives. These methodseffectively shows the balance between the optimal objectives, but at the cost of a large number of
calculations and computation (Yu et al., 2020). Time series forecasting are an active research area,
which have received a considerable amount of attention in the literature (Rozenberg et al., 2012). In
parametric design environments, the use of ANNs promises greater feasibility than simulations for
exploring the performance of different solutions, due to a reduction in overall computation time
(Lorenz et al., 2020).

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
