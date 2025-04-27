import torch
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class AttentionLSTM(torch.nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2):
        super(AttentionLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = torch.nn.Linear(hidden_size, 1)
        self.fc = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.softmax(self.attention(lstm_out))
        context = torch.sum(lstm_out * attention_weights, dim=1)
        output = self.fc(context)
        return output, attention_weights

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lstm(train_loader, val_loader, device, epochs=10):
    model = AttentionLSTM().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output, _ = model(batch_x)
                val_loss += criterion(output, batch_y).item()
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader):.4f}')
    return model

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    if isinstance(model, torch.nn.Module):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 64):
                batch_x = torch.tensor(X_test[i:i+64], dtype=torch.float32).to(model.device)
                output, _ = model(batch_x)
                batch_preds = torch.argmax(output, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        preds = np.array(preds)
    else:
        preds = model.predict(X_test)
    print(f"\n{model_name} Classification Report:")
    report = classification_report(y_test, preds, target_names=['Non-FOG', 'FOG'], zero_division=0)
    print(report)
    if feature_names and not isinstance(model, torch.nn.Module):
        importances = model.feature_importances_ if model_name in ['Random Forest', 'XGBoost'] else np.abs(model.coef_[0])
        print(f"\n{model_name} Feature Importance:")
        for name, imp in zip(feature_names, importances):
            print(f"{name}: {imp:.4f}")
        plt.figure(figsize=(10, 8))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
        plt.close()

def evaluate_on_normal(models, normal_ml_data, normal_loaders, feature_names, device):
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name} on Normal Data...")
        all_preds, all_labels = [], []
        fog_pred_data = []
        if name == 'AttentionLSTM':
            model.eval()
            with torch.no_grad():
                for loader, subject in normal_loaders:
                    preds, labels = [], []
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(device)
                        output, _ = model(batch_x)
                        batch_preds = torch.argmax(output, dim=1).cpu().numpy()
                        preds.extend(batch_preds)
                        labels.extend(batch_y.numpy())
                        if np.any(batch_preds == 1):
                            fog_pred_data.append((batch_x.cpu().numpy()[batch_preds == 1], subject))
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    print(f"Subject {subject}: {np.unique(preds, return_counts=True)}")
        else:
            for X_norm_ml, y_norm_win, subject in normal_ml_data:
                preds = model.predict(X_norm_ml)
                all_preds.extend(preds)
                all_labels.extend(y_norm_win)
                if np.any(preds == 1):
                    fog_pred_data.append((X_norm_ml[preds == 1], subject))
                print(f"Subject {subject}: {np.unique(preds, return_counts=True)}")
        print(f"{name} Labels: {np.unique(all_labels, return_counts=True)}")
        print(f"{name} Predictions: {np.unique(all_preds, return_counts=True)}")
        report = classification_report(
            all_labels, all_preds, 
            labels=[0, 1], 
            target_names=['Normal', 'FOG'], 
            output_dict=True, 
            zero_division=0
        )
        results[name] = report
        print(f"\n{name} Classification Report:")
        print(pd.DataFrame(report).transpose())
        if fog_pred_data and name != 'AttentionLSTM':
            print(f"\nComputing SHAP for {name} FOG predictions...")
            explainer = shap.TreeExplainer(model) if name in ['Random Forest', 'XGBoost'] else shap.LinearExplainer(model, X_norm_ml)
            for X_fog, subject in fog_pred_data:
                print(f"FOG data shape for {subject}: {X_fog.shape}")
                shap_values = explainer.shap_values(X_fog)
                shap_values_fog = shap_values if name in ['XGBoost', 'Logistic Regression'] else shap_values[1]
                shap_values_fog = np.array(shap_values_fog)
                if shap_values_fog.ndim == 1 and shap_values_fog.size == X_fog.shape[1]:
                    shap_values_fog = shap_values_fog.reshape(1, -1)
                if shap_values_fog.shape[0] == X_fog.shape[0] and shap_values_fog.shape[0] > 0:
                    shap.summary_plot(shap_values_fog, X_fog, feature_names=feature_names, show=False)
                    plt.title(f"SHAP for FOG Predictions ({name}, {subject})")
                    plt.savefig(f"shap_{name.lower().replace(' ', '_')}_{subject}.png")
                    plt.close()

    return results