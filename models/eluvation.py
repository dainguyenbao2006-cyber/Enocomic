import numpy as np


def r2_score(y_true, y_pred):
    """
    Hàm tính toán R-squared score từ đầu.
    
    Tham số:
    y_true: list hoặc numpy array chứa các giá trị thực tế.
    y_pred: list hoặc numpy array chứa các giá trị dự đoán từ mô hình.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    

    y_mean = np.mean(y_true)
    

    ss_res = np.sum((y_true - y_pred) ** 2)
    

    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def mean_squared_error(y_true, y_pred):
    """
    Hàm tính toán Mean Squared Error (MSE) từ đầu.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Hàm tính toán Mean Absolute Error (MAE) từ đầu.
    """
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """
    Hàm tính toán Root Mean Squared Error (RMSE) từ đầu.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))