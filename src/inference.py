import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model import MultitaskGWASModel

def load_model(model_path='/final/best_multitask_model.pt'):
    model = MultitaskGWASModel(input_dim=31, hidden_dim=128, num_tasks=3)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.set_train_mode(False)

    return model

def preprocess_data(data_path, scaler=None):
    data = pd.read_csv(data_path)

    features = []
    for disease in ['cardio', 't2d', 'cancer']:
        features.extend([f'log_odds_{disease}', f'log_odds_se_{disease}', f'pvalue_{disease}'])

    chromosome_dummies = pd.get_dummies(data['chromosome'], prefix='chr')
    data = pd.concat([data, chromosome_dummies], axis=1)
    features.extend(chromosome_dummies.columns.tolist())

    X = data[features].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, data, scaler

def predict_risk(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(X_tensor)
    risk_scores = predictions.numpy()
    binary_predictions = (risk_scores > 0.5).astype(int)

    return risk_scores, binary_predictions


if __name__=="__main":
    # Example usage
    model = load_model()
    X, data, scaler = preprocess_data('/final/improved_multitask_gwas_data.csv')
    risk_scores, binary_predictions = predict_risk(model, X)

    results = pd.DataFrame({
        'SNP_ID': data['SNP_ID'],
        'Cardio_Risk_Score': risk_scores[:, 0],
        'T2D_Risk_Score': risk_scores[:, 1],
        'Cancer_Risk_Score': risk_scores[:, 2],
        'Cardio_High_Risk': binary_predictions[:, 0],
        'T2D_High_Risk': binary_predictions[:, 1],
        'Cancer_High_Risk': binary_predictions[:, 2]
    })