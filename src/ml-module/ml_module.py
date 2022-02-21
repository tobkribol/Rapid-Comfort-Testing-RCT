import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def read_file(file):
    # Open file and read content
    f = open(file, "r")
    x = []
    for line in f.readlines():
        for elem in line.split(","):
            x.append(float(elem))
    return x

def save_file(file, data):
    # save file
    f = open(file, "w")
    for line in data:
        for elem in line:
            f.write(str(elem)+", ")
        f.write('\n')
    f.close()
    print("fil lagret")

def normalize_input(input):
    #Normalize data
    def normalize(entry):
        return [entry[0]/120.0, entry[1]/120.0, entry[2]/360.0]

    return list(map(normalize, input))

def chunks(lst, n):
    # split data into variablegroups
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train_val_test_split(input, output, train_size, val_size, test_size, random_state=10):
    # split val, test and train data
    X_train, X_val, y_train, y_val = train_test_split(input, output,
    test_size=(val_size+test_size), random_state=random_state, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
    test_size=(test_size/(val_size+test_size)), random_state=random_state, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test



class Model(nn.Module):
    #Define network
    def __init__(self, n_in, n_out):
        super(Model, self).__init__()
        # Layers for network (in/out)
        self.fc_in = nn.Linear(n_in, n_layer0)
        self.bn_in = nn.BatchNorm1d(num_features=n_layer0)
        self.fc_hidden1 = nn.Linear(n_layer0, n_layer1)
        self.bn_hidden1 = nn.BatchNorm1d(num_features=n_layer1)
        self.fc_hidden2 = nn.Linear(n_layer1, n_layer2)
        self.bn_hidden2 = nn.BatchNorm1d(num_features=n_layer2)
        self.fc_hidden3 = nn.Linear(n_layer2, n_layer3)
        self.bn_hidden3 = nn.BatchNorm1d(num_features=n_layer3)
        self.fc_hidden4 = nn.Linear(n_layer3, n_layer4)
        self.bn_hidden4 = nn.BatchNorm1d(num_features=n_layer4)
        self.fc_out = nn.Linear(n_layer4, n_out)


    def forward(self, x):
        #Define hidden layers #F.ReLU()
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = F.relu(self.bn_hidden1(self.fc_hidden1(x)))
        x = F.relu(self.bn_hidden2(self.fc_hidden2(x)))
        x = F.relu(self.bn_hidden3(self.fc_hidden3(x)))
        x = F.relu(self.bn_hidden4(self.fc_hidden4(x)))
        x = self.fc_out(x)
        return x



if __name__ == '__main__':
    # Check for CUDA else CPU
    use_cuda = torch.cuda.is_available()
    print("Has Cuda: ", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Model init
    model = Model(5, 8760).to(device)

    #Train existing model
    eks_model = False
    if eks_model:
    model.load_state_dict(torch.load("folder/modelname.pth"))
    model.train()
    print("Use eks. model: ", eks_model)

    #Def loss and momentum
    loss_function = nn.MSELoss()
    momentum = 0.99
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # System
    batch_size = int(64)
    num_epochs = 250

     params = {
        'batch_size': batch_size, # vanlig med 16, 32, 64 x**2
        'shuffle': True,
        'num_workers': 0,
        'drop_last': False
        }

     #Hente Treningsdata
    folder = "folder"
    variabler = list(chunks(read_file(folder+"variabler.txt"), 7))
    resultater = list(chunks(read_file(folder+"resultater.txt"), 8760))

    #Del items if needed
    new_variabler = []
    for item in variabler:
    new_variabler.append(item[:])
    variabler = new_variabler
    
    #Read dimentions
    print(f'variabler lastet inn: {len(variabler), len(variabler[0])}')
    print(f'resultater lastet inn: {len(resultater), len(resultater[0])}')

    #Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        variabler, resultater, 0.75, 0.15, 0.15, 10)
    print("Split: Train, val , test:", len(X_train), len(X_val), len(X_test))

    # Convert X and Y to dataset ("TensorDataset") and extract dataloader ("DataLoader")
    Dataloader_train = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32,
        device=device), torch.tensor(y_train, dtype=torch.float32, device=device)), **params)
    Dataloader_val = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32,
        device=device), torch.tensor(y_val, dtype=torch.float32, device=device)), **params)
    Dataloader_test = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32,
        device=device), torch.tensor(y_test, dtype=torch.float32, device=device)), **params)
        
    #Generate batch size
    batches_in_epoch_train = len(X_train) / batch_size
    batches_in_epoch_val = len(X_val) / batch_size
    batches_in_epoch_test = len(X_test) / batch_size
    print(batches_in_epoch_train, batches_in_epoch_val, batches_in_epoch_test, batch_size)

    # Models
    ANN_models = [[5, 1, 1, 1, 1], [2, 2, 2, 2, 2], [8, 6, 4, 2, 1], [1, 2, 4, 6, 8],
    [6, 2, 1, 2, 6]]
    for model in len(ANN_models):
    n_layer0, n_layer1, n_layer2, n_layer3, n_layer4 = ANN_models[model]
    
    # Generate Logs
    log_loss = []
    log_epoch_loss_e = []
    log_CVRMSE = []
    log_NMBE = []
    log_acc = []
    logg_PCC = []
    log_momentum = []
    log_lr = []

    #Training
    krav = 1
    CVRMSE = 10000
    accuracy = 0.0
    PCC = 0.0
    epoch_val_loss = float(len(X_val))
    for epoch in range(num_epochs):
        # Each epoch
        if CVRMSE <= 1.0 or PCC >=90.0:
        #Train until criteria is met
        break

    else:

        if PCC > krav and epoch > 3:
        krav += 1
        momentum -= momentum/25
        lr -= lr/50
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    batch_loss = 0.0
    epoch_loss = 0.0
    batch_størrelse = 0

    for input, ground_truth in Dataloader_train:

        input = input.to(device)
        ground_truth = ground_truth.to(device)
        #Itterate each batch
        optimizer.zero_grad()
        output = model(input)
        loss = torch.sqrt_(loss_function(output, ground_truth))
        log_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        #Log_loss
        epoch_loss += (loss.item() * output.shape[0])

    if (epoch+1)%1==0:
        #Report progress
        epoch_val_loss = 0.0
        epoch_acc = 0.0
        correct = 0.0
        NMBE_step = 0.0
        CVRMSE_step = 0.0
        pcc_step = 0.0

        for input_val, ground_truth_val in Dataloader_val:
            # Report validation
            input_val = input_val.to(device)
            ground_truth_val = ground_truth_val.to(device)
            #Model validation
            output_val = model(input_val)
            optimizer.step()
            val_loss = torch.sqrt_(loss_function(output_val, ground_truth_val))
            #Calc Accuracy and Error
            #Abs error threshold
            correct += (abs(output_val - ground_truth_val)).float().sum()/8760/ground_truth_val.mean()

            # Pearson Correlation Coefficient as cost function
            vx = output_val - torch.mean(output_val)
            vy = ground_truth_val - torch.mean(ground_truth_val)
            pcc_step += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

            #NMBE: Normalized Mean Bias Error
            NMBE_step += (output_valground_truth_val).float().sum()/output_val.float().sum()

            #CV(RMSE): Coefficient of Variation of the Root Mean Square Error 
            CVRMSE_step += (torch.sqrt((torch.square(output_valground_truth_val)).float().sum()/8760)/ground_truth_val.mean()) #len(ground_truth_val)
            #(pcc_step.item(), (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))).item())
            
            #RMSE: Root Mean Square Error
            epoch_val_loss += (val_loss.item() * output_val.shape[0])

            #Calc Total Accuracy and Error
            CVRMSE = 100 * CVRMSE_step / len(X_val)
            NMBE = 100 * NMBE_step / len(X_val)
            accuracy = 100 * (1-(correct /len(X_val)))
            PCC = 100 * pcc_step / batches_in_epoch_val
            #Log data
            log_CVRMSE.append(CVRMSE.item())
            log_NMBE.append(NMBE.item())
            log_acc.append(accuracy.item())
            logg_PCC.append(PCC.item())
            log_lr.append((lr*100))
            log_momentum.append(momentum*100)
            print("Epoch {}/{}, Loss: {:.3f}, Loss_val: {:.3f}, CVRMSE: {:.2f}, NMBE: {:.1f}, Accuracy: {:.1f}, PCC: {:.1f}, Lr: {:.2f}, Momentum: {:.0f}".format(epoch + 1, num_epochs, epoch_loss / len(X_train),epoch_val_loss/ len(X_val), CVRMSE, NMBE, accuracy,PCC, lr*100, momentum*100))

        #Report log loss
        log_epoch_loss_e.append(epoch_loss/len(X_train))

    pcc_step = 0.0
    epoch_test_loss = 0.0
    epoch_acc = 0.0
    correct = 0.0
    NMBE_step = 0.0
    CVRMSE_step = 0.0
    for input_test, ground_truth_test in Dataloader_test:
        # Report Testing
        input_test = input_test.to(device)
        ground_truth_test = ground_truth_test.to(device)
        output_test = model(input_test)
        optimizer.step()
        test_loss = torch.sqrt_(loss_function(output_test, ground_truth_test))

        # Calculate Accuracy og Error
        correct += (abs(output_test -
        ground_truth_test)).float().sum()/8760/ground_truth_test.mean()
        NMBE_step += (output_test - ground_truth_test).float().sum() /
        output_test.float().sum()
        CVRMSE_step += torch.sqrt(torch.square((output_test -
        ground_truth_test)).float().sum() / 8760) / ground_truth_test.mean()
        epoch_test_loss += (test_loss.item() * output_test.shape[0])
        
        # Pearson Correlation Coefficient as cost function
        vx = output_test - torch.mean(output_test)
        vy = ground_truth_test - torch.mean(ground_truth_test)
        pcc_step += torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) *
        torch.sqrt(torch.sum(vy ** 2)))

        # Calculate loss og acc
    CVRMSE = 100 * CVRMSE_step / len(X_test)
    NMBE = 100 * NMBE_step / len(X_test)
    accuracy = (1-(correct /len(X_test)))*100
    PCC = 100 * pcc_step / batches_in_epoch_test
    # log data
    log_CVRMSE.append(CVRMSE.item())
    log_NMBE.append(NMBE.item())
    log_acc.append(accuracy.item())

    print("Test, Loss_test: {:.3f}, CVRMSE: {:.1f}, NMBE: {:.1f}, Accuracy: {:.1f}, PCC: {:.1f}".format(epoch_test_loss/len(X_test), CVRMSE, NMBE, accuracy, PCC))

    #Save model
    torch.save(model.state_dict(), "folder/fullptThermmodell.pth")
    print("done :)")