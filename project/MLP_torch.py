import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import copy
import joblib

# -----------------------------
# Funciones auxiliares
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_and_preprocess(csv_path, target_column="Calories"):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import joblib

    # Cargar DataFrame
    df = pd.read_csv(csv_path)

    # Eliminar columnas inútiles como 'Unnamed: 0' o 'id'
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop(columns=["id"], errors="ignore")

    # Procesar columna Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"male": 1, "female": 0}).astype("float32")

    # Separar X y y
    X = df.drop(columns=[target_column], errors="ignore").astype("float32").values
    y = df[target_column].astype("float32").values.reshape(-1, 1)   # <--- IMPORTANTE

    # Escalar X con MinMax
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Escalar y con StandardScaler (mejor para regresión)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y).flatten()

    # Guardar scalers
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    return df, X_scaled, y_scaled

def create_tensors(X_train, X_test, y_train, y_test, device):
    x_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32, device=device)
    return x_train, x_test, y_train, y_test

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, p_dropout=0.3):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=p_dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)         # [batch, 1]
        x = x.squeeze(-1)       # [batch]
        return x


# -------------------------------------------
# ENTRENAMIENTO CON EARLY STOPPING CORRECTO
# -------------------------------------------
def train_model(
    mlp, train_loader, val_loader,
    loss_fn, optimizer,
    epochs=1000, paciencia=100, delta=0.001
):
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(mlp.state_dict())
    no_improvement_count = 0

    for epoch in range(epochs):
        mlp.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch_x)     # [batch]
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # -----------------------
        # VALIDACIÓN
        # -----------------------
        mlp.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = mlp(batch_x)
                loss = loss_fn(outputs, batch_y)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # -----------------------
        # EARLY STOPPING CORRECTO
        # -----------------------
        if epoch_val_loss < best_val_loss - delta:
            best_val_loss = epoch_val_loss
            best_state_dict = copy.deepcopy(mlp.state_dict())
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= paciencia:
            print(f"Early stopping en epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    # cargar pesos del mejor modelo
    mlp.load_state_dict(best_state_dict)
    return mlp, train_losses, val_losses