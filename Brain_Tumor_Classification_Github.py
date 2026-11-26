import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

data = pd.read_csv("") #File Directory Here
encode = LabelEncoder()
data["label"] = encode.fit_transform(data["label"])

x = data.drop(columns = ["patient_id","label"])
y = data["label"]
patients = data["patient_id"] 

#Training mechanics
rules = StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = 7)

#Function that automates the OVA evaluation for all three classes of Logistic Regression
def logreg(Y):
    data["Class_label"] = encode.fit_transform(y==Y)
    model = Pipeline([
        ("Z-Score", StandardScaler()),
        ("Model", LogisticRegression( penalty = None, random_state =7, max_iter = 1000, class_weight="balanced"))
        ])
    y_pred = cross_val_predict(model, x, data["Class_label"], groups=patients, cv = rules)
    results = classification_report(data["Class_label"],y_pred)
    c_matrix = confusion_matrix(data["Class_label"],y_pred)
    print(f"Confusion Matrix for Logistic Regression Model {Y}")
    print(c_matrix)
    
#Head-to-Head vs FNN
class FNN():
    def __init__(self, inp, hid):
        self.w1 = np.random.randn(inp,hid)
        self.b1 = np.zeros((1,hid))
        self.w2 = np.random.randn(hid,1)
        self.b2 = np.zeros((1,1))
    
    def Feedforward(self,X):
        #Activation of hidden layer (ReLU)
        self.z1 = np.dot(X,self.w1)+self.b1
        self.a1 = np.maximum(0,self.z1)
        
        #Activation of Output layer (Sigmoid)
        self.z2 = np.dot(self.a1,self.w2)+self.b2
        self.a2 = 1/(1+np.exp(-self.z2))
        return self.a2
        
    def Loss(self, y_true, y_pred):
        err = 1e-15
        y_pred = np.clip(self.a2, err, 1-err)
        #Loss Function: Binary Cross Entropy which is mathematically identical to NLL
        loss = -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        return loss
    
    def Back(self,X,y_true):
        #Gradients of Output Layer
        "Backpropogation for 1 Layer: dL/dw = dL/da * da/dz * dz/dw"
        alpha = 0.01
        m = X.shape[0]
        dz2 = self.a2 - y_true
        dw2 = np.dot(self.a1.T, dz2)/m 
        db2 = np.sum(dz2, axis = 0, keepdims = True)/m
        
        #Gradients of Hidden Layer
        da1 = np.dot(dz2,self.w2.T)
        dz1 = da1*(self.z1>0)
        dw1 = np.dot(X.T,dz1)/m
        db1 = np.sum(dz1, axis = 0, keepdims = True)/m
        
        #Gradient Descent
        self.w1 -= dw1*alpha
        self.b1 -= db1*alpha
        self.w2 -= dw2*alpha
        self.b2 -= db2*alpha

    def model_fit(self, X, y_true, epochs = 1000):
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1,1)
        for _ in range(epochs):
            self.Feedforward(X)
            self.Back(X,y_true)
    def predict(self,X):
        return (self.Feedforward(X)>= 0.5).astype(int).ravel()

def run_FNN(Y):
    label = encode.fit_transform(y == Y)
    pred, true = [],[]
    #Introduce the looping mechanism to train and test across 5 Partitions
    for tr, te in rules.split(x,label, groups = patients):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x.iloc[tr])
        x_test = scaler.transform(x.iloc[te])
        fnn = FNN(inp= x.shape[1], hid = 16)
        fnn.model_fit(x_train, label[tr], epochs = 1000)
        
        pred.extend(fnn.predict(x_test))
        true.extend(label[te])
    print(f"FNN (1 Layer) Class {Y} Results:")
    print(confusion_matrix(true, pred))

#This should loop through all logistic regression classes 0,1,2 as well as the different regularizations of a Logistic Regression Model
for i in range(0,3):
    logreg(i)
    run_FNN(i)
