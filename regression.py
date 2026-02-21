
#Importar librerias
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Cargar el dataset
df = pd.read_csv("insurance.csv")

#Al tener valores de texto, se deben cambiar 0 y 1 para que lo entienda el modelo
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

#Definir X y y
X = df.drop("charges", axis=1).values
y = df["charges"].values.reshape(-1, 1)

#El split del train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Convertir a tensors de torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#Definir la clase de regresion (la del profe)
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

#Creacion del modleo
model = LinearRegression(X_train.shape[1], 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

#Se hace loop para el training
for epoch in range(1000):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #Imprimir cada 100
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

#Evaluacion 
with torch.no_grad():
    y_pred_test = model(X_test)
    mse = mean_squared_error(y_test.numpy(), y_pred_test.numpy())

print("Final MSE:", mse)

#Guardar reporte en un df 
report = pd.DataFrame({
    "Actual": y_test.numpy().flatten(),
    "Predecida": y_pred_test.numpy().flatten()
})

report.to_csv("reporte.csv", index=False)
print("Reporte creado")
