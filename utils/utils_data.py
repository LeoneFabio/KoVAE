"""
data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine datasets
(2) real_data_loading: Load and preprocess real datasets
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import os
import torch
import controldiffeq
import pathlib
from utils.utils import device_available
import csv
import json
import pandas as pd


PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent


def to_tensor(data):
    return torch.from_numpy(data).float()


def MinMaxScaler(data, return_minmax=False):
    """Min Max normalizer.

    Args:
      - datasets: original datasets

    Returns:
      - norm_data: normalized datasets
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    if return_minmax:
        return norm_data, min, max
    return norm_data

def inverse_MinMaxScaler(norm_data, min_data, max_data):
    return norm_data * (max_data - min_data + 1e-7) + min_data


def pendulum_nonlinear(num_points, noise, theta=2.4):
    from matplotlib import pylab as plt
    from scipy.special import ellipj, ellipk
    np.random.seed(1)

    def sol(t, theta0):
        S = np.sin(0.5 * (theta0))
        K_S = ellipk(S ** 2)
        omega_0 = np.sqrt(9.81)
        sn, cn, dn, ph = ellipj(K_S - omega_0 * t, S ** 2)
        theta = 2.0 * np.arcsin(S * sn)
        d_sn_du = cn * dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
        return np.stack([theta, d_theta_dt], axis=1)

    anal_ts = np.arange(0, 170 * 0.1, 0.1)
    # Generate random angles in radians
    # angles = np.random.uniform(1, 3.5, num_points)
    angles = np.random.uniform(.5, 2.7, num_points)
    X = []
    for theta in angles:
     X.append(sol(anal_ts, theta))

    # X = X.T
    # Xclean = X.copy()
    X = np.array(X)
    X += np.random.standard_normal(X.shape) * noise


    X = MinMaxScaler(X)

    return X

def sine_data_generation(no, seq_len, dim):
    """Sine datasets generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - datasets: generated datasets
    """
    # Initialize the output
    data = list()

    # Generate sine datasets
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated datasets
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len, return_minmax=False):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock, energy or Other datasets
      - seq_len: sequence length

    Returns:
      - datasets: preprocessed datasets.
    """
    assert data_name in ['stock', 'energy', 'metro', 'EV']
    
    if data_name == 'stock':
        csv_path = './datasets/stock_data.csv'
    elif data_name == 'energy':
        csv_path = './datasets/energy_data.csv'
    elif data_name == 'metro':
        csv_path = './datasets/metro_data.csv'
    elif data_name == 'EV':
        csv_path = './datasets/EV_data.csv'

    # Read the full CSV, header included
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # First row = feature names
        ori_data = np.loadtxt(f, delimiter=",")

    # Save features (columns' names) in the directory ./Output_info if not exist
    output_dir = './Output_info'
    file_path = os.path.join(output_dir, f'{data_name}_features.json')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(file_path, 'w') as f:
            json.dump(header, f)

    if data_name == 'EV':
        # Do NOT flip the dataset
        # Normalize the dataset
        ori_data, min_data, max_data = MinMaxScaler(ori_data, return_minmax=True)

        # Preprocess the dataset using the sliding window
        temp_data = []
        # Cut dataset by sequence length
        for i in range(0, len(ori_data) - seq_len):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)

        # Do NOT mix the dataset
        # data is the variable to return
        data = temp_data

    else:
        #NOT EV dataset

        # Flip the datasets to make chronological datasets
        ori_data = ori_data[::-1]
        # Normalize the datasets
        ori_data = MinMaxScaler(ori_data)

        # Preprocess the datasets
        temp_data = []
        # Cut datasets by sequence length
        for i in range(0, len(ori_data) - seq_len):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        data = []
        for i in range(len(temp_data)):
            data.append(temp_data[idx[i]])

    if return_minmax:
        # Return the normalized datasets and the min/max values
        return data, min_data, max_data
    return data

def preprocess_dataset(path, data_name):

    # Load CSV
    df = pd.read_csv(path, sep=',')

    # Drop the first column (vin -> Vehicle's identifier)
    df = df.iloc[:, 1:]

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], dayfirst=True, errors='coerce')

    # Compute event duration in minutes
    df['duration_minutes'] = (df['end_time'] - df['timestamp']).dt.total_seconds() / 60

    # Drop original time columns
    df = df.drop(columns=['timestamp', 'end_time'])
    
    ''' Label encoding 
    
    # Encode 'event': trip → 0, charge → 1
    event_mapping = {'trip': 0, 'charge': 1}
    df['event'] = df['event'].map(event_mapping)

    # Encode 'charge_mode': NaN → 0, '240' → 1, 'DC Charging' → 2
    def encode_charge_mode(val):
        if pd.isna(val):
            return 0
        elif str(val).strip() == '240':
            return 1
        elif 'DC' in str(val):
            return 2
        else:
            return 0  # fallback for unexpected values

    df['charge_mode'] = df['charge_mode'].apply(encode_charge_mode)'''
    
    """One-hot encoding"""
    # Normalize/clean categorical columns
    df['event'] = df['event'].str.strip().str.lower()
    df['charge_mode'] = df['charge_mode'].astype(str).str.strip()

    # One-hot encode 'event' (trip, charge)
    df = pd.get_dummies(df, columns=['event'], prefix='event')

    # Map 'charge_mode' to categorical labels first
    def map_charge_mode(val):
        if pd.isna(val) or val.lower() == 'nan':
            return 'none'
        elif val == '240':
            return '240'
        elif 'DC' in val.upper():
            return 'dc'
        else:
            return 'none'  # fallback for unexpected values

    df['charge_mode'] = df['charge_mode'].apply(map_charge_mode)

    # One-hot encode 'charge_mode' (none, 240, dc)
    df = pd.get_dummies(df, columns=['charge_mode'], prefix='charge')
    
    
    
    ######################## HANDLE MISSING VALUES ######################################
    # Fill any remaining NaNs (e.g., missing soc values) 
    df = df.fillna(method='ffill')
    ######################## KEEP STUDYING THE BEST OPTION ##############################
    
    # Ensure all values are float-compatible (0/1 instead of True/False)
    df = df.astype(float)
    

    # Save the processed dataset
    output_dir = './Output_info'
    file_path = os.path.join(output_dir, f'{data_name}_processed.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(file_path, index=False)
    
    print(f'Preprocessing complete. Output saved to {file_path}')
    
    return file_path

def decode_categorical_from_generated(generated_data):
            """
            Decode one-hot encoded categorical features (event and charge_mode) 
            from generated data and reconstruct the dataset with decoded integer labels.

            Consider:
            - generated_data shape is (batch_size, seq_len, n_features)
            - event one-hot columns at indices 2 and 3
            - charge_mode one-hot columns at indices 4, 5, 6
            """

            # Indices of one-hot encoded categorical features
            event_cols = [2, 3]
            charge_cols = [4, 5, 6]

            # Extract event one-hot predictions and decode via argmax along last axis
            event_preds = generated_data[:, :, event_cols]  # shape (batch, seq_len, 2)
            event_decoded = np.argmax(event_preds, axis=2)  # shape (batch, seq_len)

            # Extract charge_mode one-hot predictions and decode via argmax
            charge_preds = generated_data[:, :, charge_cols]  # shape (batch, seq_len, 3)
            charge_decoded = np.argmax(charge_preds, axis=2)  # shape (batch, seq_len)

            # Create a mask to keep all columns except the one-hot encoded ones
            mask = np.ones(generated_data.shape[2], dtype=bool)
            mask[event_cols + charge_cols] = False

            # Extract the features without the one-hot encoded columns
            data_wo_one_hot = generated_data[:, :, mask]  # shape (batch, seq_len, remaining_features)

            # Now append the decoded categorical columns at the end
            # decoded_cats shape should be (batch, seq_len, 2)
            decoded_cats = np.stack((event_decoded, charge_decoded), axis=2)

            # Concatenate along the last axis (features)
            processed_data = np.concatenate((data_wo_one_hot, decoded_cats), axis=2)

            return processed_data, event_decoded, charge_decoded

class TimeDataset_irregular(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0, return_minmax=False):
        SEED = 56789
        base_loc = PROJECT_DIR / 'datasets'
        loc = PROJECT_DIR / 'datasets' / (data_name + str(missing_rate))
        # if data is in cache
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

            if return_minmax:
                self.min_data = tensors.get('min_data')
                self.max_data = tensors.get('max_data')
                if self.min_data is None or self.max_data is None:
                    print(f"[Warning] min/max data not found in cache for {data_name}.")

        else:  # preprocess data according to missing rate
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(loc):
                os.mkdir(loc)
                
            self.min_data = 0
            self.max_data = 0

            if data_name == 'EV':
                # EV dataset

                csv_path = f'./datasets/{data_name}_data.csv'
                processed_dataset_path = preprocess_dataset(csv_path, data_name)
                # Read the full CSV, header included
                with open(processed_dataset_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # First row = feature names
                    print(header)
                    data = np.loadtxt(f, delimiter=",")

                # Save features (columns' names) in the directory ./Output_info if not exist
                output_dir = './Output_info'
                file_path = os.path.join(output_dir, f'{data_name}_features.json')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(file_path, 'w') as f:
                    json.dump(header, f)
                
                # Do NOT flip the dataset
                # Normalize the dataset
                norm_data, min_data, max_data = MinMaxScaler(data, return_minmax=True)
                if return_minmax:
                    self.min_data = min_data
                    self.max_data = max_data

                total_length = len(norm_data)
                time = np.array(range(total_length)).reshape(-1, 1)
                

                self.original_sample = []
                ori_seq_data = []

                for i in range(len(norm_data) - seq_len + 1):
                    x = norm_data[i: i + seq_len].copy()
                    ori_seq_data.append(x)

                self.original_sample = ori_seq_data.copy()
                orig_samples_np = np.array(self.original_sample)
                self.X_mean = np.mean(orig_samples_np, axis=0).reshape(1, orig_samples_np.shape[1], orig_samples_np.shape[2])

                # Do NOT mix the dataset
                '''idx = torch.randperm(len(ori_seq_data))
                for i in range(len(ori_seq_data)):
                    self.original_sample.append(ori_seq_data[idx[i]])
                orig_samples_np = np.array(self.original_sample)
                self.X_mean = np.mean(orig_samples_np, axis=0).reshape(1, orig_samples_np.shape[1], orig_samples_np.shape[2])'''

                generator = torch.Generator().manual_seed(SEED)
                removed_points = torch.randperm(norm_data.shape[0], generator=generator)[
                                :int(norm_data.shape[0] * missing_rate)].sort().values
                norm_data[removed_points] = float('nan')
                norm_data = np.concatenate((norm_data, time), axis=1)
                seq_data = []
                for i in range(len(norm_data) - seq_len + 1):
                    x = norm_data[i: i + seq_len]
                    seq_data.append(x)
                self.samples = seq_data.copy()
                '''for i in range(len(seq_data)):
                    self.samples.append(seq_data[idx[i]])'''
            else:
                # NOT EV dataset
                
                if data_name in ['stock', 'energy']:
                    csv_path = f'./datasets/{data_name}_data.csv'
                    # Read the full CSV, header included
                    with open(csv_path, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader)  # First row = feature names
                        data = np.loadtxt(f, delimiter=",")

                    # Save features (columns' names) in the directory ./Output_info if not exist
                    output_dir = './Output_info'
                    file_path = os.path.join(output_dir, f'{data_name}_features.json')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(file_path, 'w') as f:
                        json.dump(header, f)
                    
                    data = data[::-1]
                    norm_data= MinMaxScaler(data)
                    total_length = len(norm_data)
                    time = np.array(range(total_length)).reshape(-1, 1)
                elif data_name == 'mujoco':
                    tensors = load_data(loc)
                    time = tensors['train_X'][:, :, :1].cpu().numpy()
                    data = tensors['train_X'][:, :, 1:].reshape(-1, 14).cpu().numpy()
                    norm_data = MinMaxScaler(data)
                    norm_data = norm_data.reshape(4620, seq_len, 14)
                elif data_name == 'sine':
                    norm_data = sine_data_generation(no=10000, seq_len=24, dim=5)

                self.original_sample = []
                ori_seq_data = []

                for i in range(len(norm_data) - seq_len + 1):
                    x = norm_data[i: i + seq_len].copy()
                    ori_seq_data.append(x)
                idx = torch.randperm(len(ori_seq_data))
                for i in range(len(ori_seq_data)):
                    self.original_sample.append(ori_seq_data[idx[i]])
                orig_samples_np = np.array(self.original_sample)
                self.X_mean = np.mean(orig_samples_np, axis=0).reshape(1, orig_samples_np.shape[1], orig_samples_np.shape[2])

                generator = torch.Generator().manual_seed(SEED)
                removed_points = torch.randperm(norm_data.shape[0], generator=generator)[
                                :int(norm_data.shape[0] * missing_rate)].sort().values
                norm_data[removed_points] = float('nan')
                norm_data = np.concatenate((norm_data, time), axis=1)
                seq_data = []
                for i in range(len(norm_data) - seq_len + 1):
                    x = norm_data[i: i + seq_len]
                    seq_data.append(x)
                self.samples = []
                for i in range(len(seq_data)):
                    self.samples.append(seq_data[idx[i]])

            #From here, the data is ready (either EV or not)
            self.samples = np.array(self.samples)

            device = device_available()
            norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().to(device)

            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).to(device)
            self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            

            save_data(loc, data=self.samples,
                      original_data=self.original_sample,
                      train_a=self.train_coeffs[0],
                      train_b=self.train_coeffs[1],
                      train_c=self.train_coeffs[2],
                      train_d=self.train_coeffs[3],
                      min_data=self.min_data,
                      max_data=self.max_data,
                      )

            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, index):
        batch_coeff = (self.train_coeffs[0][index].float(),
                       self.train_coeffs[1][index].float(),
                       self.train_coeffs[2][index].float(),
                       self.train_coeffs[3][index].float())

        self.sample = {'data': self.samples[index], 'inter': batch_coeff, 'original_data': self.original_sample[index]}

        return self.sample  # self.samples[index]

    def __len__(self):
        return len(self.samples)

def create_timeDataset_irregular(data_name, seq_len, missing_rate=0.0, return_minmax=False):
    """Create time-series irregular dataset instance

    Args:
      - data_name: stock, energy or Other datasets
      - seq_len: sequence length
      - missing_rate: the rate of missing values
      - return_minmax: whether to return min/max values useful for de-normalize

    Returns:
      - dataset: preprocessed dataset
      - min_data: min values of the dataset
      - max_data: max values of the dataset
    """
    dataset = TimeDataset_irregular(seq_len, data_name, missing_rate=missing_rate, return_minmax=return_minmax)    
    if return_minmax:
        return dataset, dataset.min_data, dataset.max_data
    else:
        return dataset

def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')