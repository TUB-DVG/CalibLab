import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import os
try:
    import inputs 
except:
    import src.inputs as inputs
try:
    import paths 
except:
    import src.paths as paths



def train_gp(scr_gebaeude_id, num_bc_param, kernel, kernel_index, num_samples_gp, gp_test_size, output_resolution, training_ratio):

    dir_path = os.path.join(paths.DATA_DIR, "GP", str(scr_gebaeude_id), f'{output_resolution}', f'{num_bc_param}_bc_param')
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except FileExistsError:
        # Another process in the multiprocessing may have created the directory already; just pass
        pass

    # Fetching samples
    file_gp_samples = open(os.path.join(paths.DATA_DIR, f'GP/{scr_gebaeude_id}/{output_resolution}/{num_bc_param}_bc_param/Indata_GP_{training_ratio}_{num_samples_gp}_samples.pkl'), "rb")
    data = pickle.load(file_gp_samples)
    file_gp_samples.close()

    data_Y = data['heating_energy'].apply(lambda x: pd.Series(x)).values
    data_X = data.drop("heating_energy", axis=1).values

    assert num_bc_param == data_X.shape[1]  # Control

    # Split & Scale the training data
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=gp_test_size, random_state=42) 
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)  

    # Gaussian processes
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(num_samples_gp*0.1), normalize_y=True)
    gaussian_process.fit(X_train_scaled, y_train)

    # Prediction and evaluation
    X_test_scaled = scaler_X.transform(X_test)
    y_pred, y_pred_sigma = gaussian_process.predict(X_test_scaled, return_std=True)
    sigma = np.mean(np.array(y_pred_sigma))

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate Mean Avarage Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate Mean Squared Error (MSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Calculate R-squared (coefficient of determination)
    r2 = r2_score(y_test, y_pred)
    
    #Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    ###RMSE NOrmalised
    y_test_mean = np.mean(np.array(y_test))
    rmse_normalised = rmse/y_test_mean

    # Saving
    objects = [gaussian_process, scaler_X, X_train, X_test, y_train, y_test, y_pred, sigma, mse, mae, rmse, r2, mape]
    
    if output_resolution==None:
        file_gp = open(os.path.join(paths.DATA_DIR, f'GP/{scr_gebaeude_id}/{output_resolution}/{num_bc_param}_bc_param/Trained_GP_{num_samples_gp}_{kernel_index}_kernel.pkl'), "wb")
    else:
        file_gp = open(os.path.join(paths.DATA_DIR, f'GP/{scr_gebaeude_id}/{output_resolution}/{num_bc_param}_bc_param/Trained_GP_{num_samples_gp}_{training_ratio}_obs_train_{kernel_index}_kernel.pkl'), "wb")
        
    pickle.dump(objects, file_gp)
    file_gp.close()

    return gaussian_process.kernel_, mse, mae, rmse, r2, sigma, y_test_mean, mape, rmse_normalised
