# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="770" height="766" alt="image" src="https://github.com/user-attachments/assets/4aeb2858-739e-4068-9dcb-f3b6ba00b875" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:D.B.V.SAI GANESH
### Register Number: 212223240025
```python
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
sai_brain=Neuralnet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(sai_brain.parameters(),lr=0.001)

def train_model(sai_brain,x_train,y_train,criteria,optmizer,epochs=4000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(sai_brain(x_train),y_train)
        loss.backward()
        optimizer.step()
        
        sai_brain.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")



```
## Dataset Information

<img width="333" height="680" alt="image" src="https://github.com/user-attachments/assets/a001022d-4b21-4d56-947c-6e1037612a1f" />

## OUTPUT
<img width="504" height="467" alt="image" src="https://github.com/user-attachments/assets/6d1883bb-dd64-49cb-b5ad-53180ee50769" />

### Training Loss Vs Iteration Plot

<img width="823" height="582" alt="image" src="https://github.com/user-attachments/assets/d0018684-ac85-4a6a-a742-d0614eb5d6a7" />

### New Sample Data Prediction

<img width="968" height="148" alt="image" src="https://github.com/user-attachments/assets/7cdd7cbf-b82a-4422-8952-af87afb2ad8c" />

## RESULT

Successfully executed the code to develop a neural network regression model.

