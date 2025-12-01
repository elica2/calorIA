import torch
from MLP_torch import MLP

def load_model(path="best_model_regression.pth"):
    # Cargar checkpoint
    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    # Reconstruir el modelo con la misma arquitectura
    model = MLP(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=1,
        p_dropout=checkpoint["p_dropout"]
    )

    # Cargar pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # modo inferencia (importantísimo)

    return model