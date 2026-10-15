"""
Módulo de Deep Learning en PyTorch: Red Neuronal Multicapa (MLP) con Loss Asimétrica de Negocio.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
try:
    from src.models.asymmetric_loss import AsymmetricLossPyTorch, ASYMMETRY_RATIO
except ModuleNotFoundError:
    from asymmetric_loss import AsymmetricLossPyTorch, ASYMMETRY_RATIO


class FinancialMLP(nn.Module):
    """
    Arquitectura de Red Neuronal Multicapa (MLP) para datos tabulares financieros multimodales.
    Produce logits continuos directos z in R sin capa Sigmoid interna, garantizando
    estabilidad numérica en la función de pérdida asimétrica.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(FinancialMLP, self).__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())  # Swish activation function
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Logit output layer (sin Sigmoid)
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchMLPTrainer:
    """
    Clase de entrenamiento e inferencia para FinancialMLP con Asymmetric Loss PyTorch (24x).
    Incluye optimizador AdamW, programación de tasa de aprendizaje CosineAnnealingLR y
    cálculo de probabilidades calibradas en inferencia vía torch.sigmoid(logits).
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3, lr=1e-3, cost_fn=ASYMMETRY_RATIO, cost_fp=1.0, weight_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FinancialMLP(input_dim, hidden_dims, dropout_rate).to(self.device)
        self.criterion = AsymmetricLossPyTorch(cost_fn=cost_fn, cost_fp=cost_fp)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=40, batch_size=512, verbose=False):
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                # Gradient clipping para prevenir explosión de gradientes con ratio 24x
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            scheduler.step()
            avg_train_loss = total_loss / len(X_train)

            # Validación si está disponible
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.float32).unsqueeze(1).to(self.device)
                    val_logits = self.model(X_val_t)
                    val_loss = self.criterion(val_logits, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f" | Val Loss: {val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch [{epoch+1}/{epochs}] Train Asymmetric Loss: {avg_train_loss:.4f}{val_str}")

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_logits(self, X):
        """Devuelve los logits directos z in R."""
        self.model.eval()
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy().flatten()
        return logits

    def predict_proba(self, X):
        """Devuelve las probabilidades continuas calibradas p in [0, 1] via torch.sigmoid(logits)."""
        self.model.eval()
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1.0 - probs, probs]) if len(probs.shape) == 1 else probs

    def predict_proba_positive(self, X):
        """Devuelve el vector 1D de probabilidades de quiebra p(y=1|x)."""
        self.model.eval()
        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs


if __name__ == "__main__":
    print("[PyTorch MLP] Testing FinancialMLP initialization and forward pass...")
    X_dummy = np.random.randn(200, 50).astype(np.float32)
    y_dummy = np.random.randint(0, 2, size=200).astype(np.float32)

    trainer = PyTorchMLPTrainer(input_dim=50)
    trainer.fit(X_dummy, y_dummy, epochs=10, verbose=True)
    preds = trainer.predict_proba_positive(X_dummy)
    print(f"[PyTorch MLP] Success! Predicted {len(preds)} probabilities. Sample probs:", preds[:5])

