import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def getModel(X, y, threads=None):
    if threads is not None:
        torch.set_num_threads(threads)
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    dimension = X_train.shape[1]
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.layer1 = nn.Linear(dimension, 2 * dimension)
            self.layer2 = nn.Linear(2 * dimension, dimension)
            self.output_layer = nn.Linear(dimension, 1)
        
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.output_layer(x)
            return x
    
    def train_and_evaluate():
        model = NeuralNetwork()
        model.apply(init_weights)
        
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        epochs = 100
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        return model, best_loss
    
    best_model = None
    lowest_loss = float('inf')
    
    for _ in range(3):
        model, train_loss = train_and_evaluate()
        if train_loss < lowest_loss:
            lowest_loss = train_loss
            best_model = model

    return best_model

def getModelScores(model, points):
    """Return Model predictions for points"""
    t_points = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        y = torch.sigmoid(model(t_points))

    return y.numpy().squeeze()

def getModelPrediction(model, points, threshold):
    """Return Model predictions for points"""
    t_points = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        y = torch.sigmoid(model(t_points))

    return y.numpy().squeeze() >= threshold
