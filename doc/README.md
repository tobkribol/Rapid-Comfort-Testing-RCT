# data
✔️ Version: 1.0 
| 📚 Category: data
| 🛠️ Discipline: RIEn
| Submitted by: TK

## 🧰 Description 
After simulation and the parametric run, Illuminance and Operative temperature results are divided
into three datasets, referred to “2” in Figure 5. This is to ensure that the model is tested and validated
with a separate dataset. The split is based on research found in the literature and consist of 70 %
training, 15 % validation, and 15 % testing. The validation dataset is a smaller subset of the data and is
used for evaluation during training. The testing data, common referred to as the holdout dataset, is also
a smaller subset of the data, and is used for evaluation after all the training is done. This represents the
final accuracy and is used to ensure that the final model is properly generalized. This will provide the
model with stable data where it is less likely to have a bias for a certain model-case. The data is
shuffled before splitting, ensuring all datasets represents a mix of all variations including window size,
window properties and orientation. This process is illustrated in Figure 10.

![image](https://user-images.githubusercontent.com/79943730/155013723-5f2201ea-c143-4fdd-b2a3-36e0649e1ee1.png)
