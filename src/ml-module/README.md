# ml-module
✔️ Version: 1.0 
| 📚 Category: ml
| 🛠️ Discipline: RIEn
| Submitted by: TK

## 🧰 Description 
In this step the ANN is developed and trained, illustrated as numbered “3” in Figure 11. Findings from
the literature indicate that there are few tools available for conducting fully automated ANN training in 
combination with measured or simulated data. There are some plug-in tools available for Rhino and 
Grasshopper, such as LunchBoxML, Crow and Octopus, but they often lack proper detailing, or is not 
open-sourced. A custom machine learning algorithm is therefore developed in this study, using Python 
and the open source machine learning framework, Pytorch (Paszke et al., 2019).

![image](https://user-images.githubusercontent.com/79943730/155007666-a204d0af-7c90-4bb3-85e6-ac5f958c84e2.png)

## 👋 Quickstart
- Python version 3.8
- Pytorch version 1.8.1
- Anaconda Navigator version 1.10
- Cudatoolkit 10.2

## ⚙️ Setup
The ANN is developed in programming language python version 3.8 and Pytorch version 1.8.1
(Paszke et al., 2019). The environment is set up with Anaconda Navigator version 1.10, which has 
proven to be useful when working with multiple remote virtual machines, by importing custom 
existing environments directly.

ANNs require training data to learn relationships between design parameters and corresponding 
daylight illuminance and operative temperature. There is some ongoing research on automating the 
neural architecture search, which speeds up the process of developing an ANN (Kyriakides and 
Margaritis, 2021). Small networks with few numbers of hidden neurons have shown good
performance predicting temperature (Tran et al., 2020). Based on this research, a seven-layer fully 
connected ANN is used. The architecture is structured with 7 input neurons and 8760 output neurons 
using 5 hidden layers with 1 to 8 neurons in each layer. There are in total five model architectures with 
different layer structure as shown in Table 6.

![image](https://user-images.githubusercontent.com/79943730/155007371-cf20511d-ee96-40da-a345-4e26d25a0119.png)

The input variables are glass properties (LT-value and G-value), window size, window surface 
orientation, vertical view, horizontal view, view direction and distance to the window, resulting in a 
total of 7 input variables. The merged results and input variables are defined as the dataloader, which 
is used for training and validating the ANN. The training data from the dataloader is shuffled 
randomly and divided into batches with size of 64 for every epoch, the process is illustrated in Figure 
12. This ensures that every training is unique, which helps the ANN to converge faster and not fall into
a local minimum.

![image](https://user-images.githubusercontent.com/79943730/155007449-c85e2109-217f-4c70-afd1-b7c858cec2f7.png)

With the use of RMSE, the weight and biases are optimized by implementation of stochastic gradient 
descent (SGD) utilizing Nesterov momentum (Sutskever et al., 2013). This is a common optimization 
technique for training machine learning algorithms. The goal is to adjust the weights to minimize the 
RMSE loss. The gradient descent algorithm starts with random model parameters and calculates the 
error for each learning iteration. How much the weights are adjusted each iteration depends on the 
scalar hyper-parameter, known as momentum and learning rate. These hyper-parameters can be 
adaptive, meaning they change while training. Some optimizer algorithms, such as Adaptive 
Momentum Estimation (Adam) does this automatically, but often generalizes significantly worse than 
Stochastic Gradient Descent (SGD) (Xie et al., 2020, Moreira and Fiesler, 1995). We have therefore in 
this study simplified the adaptive process to change with the accuracy function. Initially the learning 
rate is set to 0.01 and momentum is set to 0.99. While training both parameters are updated depending 
on the accuracy as shown in Figure 13. When accuracy increases the momentum decreases with its 
previous value divided by 25, and learning rate decreases with its previous value divided by 50. This 
allows us to give the optimizer more impact on the adjusted weight early in the training process. 
Momentum and learning rate are logged while training.

![image](https://user-images.githubusercontent.com/79943730/155007533-e6b5c6f1-f92f-4476-ab03-fe4a01501c6d.png)

o avoid co-adaptation (overfitting) of neurons during the training process an early-stop mechanism is
implemented. This is solved by implementing batch normalization and early-stop function. After each
epoch, the algorithm measures the performance of the model over the validation data after the weigts 
and biases are updated. If the error of the validation data grows over each epoch the training is 
stopped. The validation data does not intervene in the adjustment of the weights and biases during the 
training process. Stop mechanisms are integrated while training, if CV(RMSE) decreases to less than 
1.0 %, accuracy exceeds 90 % or if the model reaches 250 epochs. For the limited budget results, 50 
epochs are used.

For training the neural network a Nvidia tesla T4 tensor core GPU with 2560 CUDA cores is used. 
Cudatoolkit 10.2 is used for enabling CUDA utilization (Cook, 2012, Vingelmann and Fitzek, 2020).
