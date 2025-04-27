import torch
from preprocess import load_daphnet_for_ml, load_daphnet_for_dl, load_normal_for_ml, load_normal_for_dl
from models import train_random_forest, train_xgboost, train_logistic_regression, train_lstm, evaluate_model, evaluate_on_normal

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_names = [f"{sensor}_{stat}" for sensor in ['Ankle_x', 'Ankle_y', 'Ankle_z', 'Thigh_x', 'Thigh_y', 'Thigh_z', 'Trunk_x', 'Trunk_y', 'Trunk_z']
                     for stat in ['mean', 'std', 'min', 'max']]
    
    train_loader, val_loader, test_loader = load_daphnet_for_dl(batch_size=64, window_len=24, step_size=12)
    X_train_ml, X_val_ml, X_test_ml, y_train_ml, y_val_ml, y_test_ml = load_daphnet_for_ml(window_len=24, step_size=12)
    
    rf_model = train_random_forest(X_train_ml, y_train_ml)
    xgb_model = train_xgboost(X_train_ml, y_train_ml)
    lr_model = train_logistic_regression(X_train_ml, y_train_ml)
    lstm_model = train_lstm(train_loader, val_loader, device)
    
    evaluate_model(rf_model, X_test_ml, y_test_ml, 'Random Forest', feature_names)
    evaluate_model(xgb_model, X_test_ml, y_test_ml, 'XGBoost', feature_names)
    evaluate_model(lr_model, X_test_ml, y_test_ml, 'Logistic Regression', feature_names)
    evaluate_model(lstm_model, X_test_ml, y_test_ml, 'AttentionLSTM')
    
    normal_files = ['data/normal/sub1.txt', 'data/normal/sub2.txt', 'data/normal/sub3.txt']  # Placeholder
    normal_ml_data = load_normal_for_ml(window_len=24, step_size=12, normal_files=normal_files)
    normal_loaders = load_normal_for_dl(batch_size=64, window_len=24, step_size=12, normal_files=normal_files)
    
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Logistic Regression': lr_model,
        'AttentionLSTM': lstm_model
    }
    
    evaluate_on_normal(models, normal_ml_data, normal_loaders, feature_names, device)

if __name__ == "__main__":
    main()