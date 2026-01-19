# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:22:03 2026

@author: willi
"""
import json
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# --- User paths ---
json_train = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\train.json"
json_test  = r"C:\Users\willi\Desktop\DTU\Master\Materials design with machine learning and AI\test.json"

def load_json_to_df(path):
    return pd.read_json(path)

def atoms_from_record(atoms_rec):
    if isinstance(atoms_rec, str):
        atoms_rec = json.loads(atoms_rec)
    numbers = atoms_rec.get("numbers")
    positions = atoms_rec.get("positions")
    cell = atoms_rec.get("cell", None)
    pbc = atoms_rec.get("pbc", None)
    tags = atoms_rec.get("tags", None)
    if cell is not None:
        atoms = Atoms(numbers=list(numbers),
                      positions=np.array(positions),
                      cell=np.array(cell),
                      pbc=tuple(pbc) if pbc is not None else False)
    else:
        atoms = Atoms(numbers=list(numbers), positions=np.array(positions))
    if tags is not None:
        atoms.set_tags(list(tags))
    return atoms

# --- Load datasets ---
df_train_full = load_json_to_df(json_train)
df_test_final = load_json_to_df(json_test)

# Increased N_train toward the full dataset for better results
N_train = 8000 
df_train_full = df_train_full.iloc[:N_train].reset_index(drop=True)

def df_to_ase_list(df):
    ase_list = []
    for idx, row in df.iterrows():
        ase_list.append(atoms_from_record(row["atoms"]))
    return ase_list

atoms_train_full = df_to_ase_list(df_train_full)
atoms_test_final = df_to_ase_list(df_test_final)

# --- Fingerprinting ---
def get_unique_species(ase_list1, ase_list2=None):
    nums = set()
    for a in ase_list1 + (ase_list2 or []):
        nums.update(a.get_atomic_numbers())
    return sorted({chemical_symbols[n] for n in nums})

# --- Advanced Feature Engineering (ASE Upgrades) ---
species = get_unique_species(atoms_train_full, atoms_test_final)

def improved_feat(a):
    # 1. Chemical Composition
    counts = Counter([chemical_symbols[n] for n in a.get_atomic_numbers()])
    comp_list = [counts.get(s, 0) for s in species]
    
    # 2. ASE Structural Features
    n_atoms = len(a)
    vol_per_atom = a.get_volume() / n_atoms
    density = np.sum(a.get_masses()) / a.get_volume()
    
    # 3. ASE Bond Length Upgrade (with safety check for single-atom systems)
    if n_atoms > 1:
        dm = a.get_all_distances(mic=True)
        # We only calculate min distance if there are neighbors
        min_dists = []
        for row in dm:
            nonzero_dists = row[row > 1e-5] # Use a small epsilon instead of 0
            if len(nonzero_dists) > 0:
                min_dists.append(np.min(nonzero_dists))
            else:
                # Fallback if an atom is isolated even in a multi-atom cell
                min_dists.append(0.0) 
        avg_bond_dist = np.mean(min_dists)
    else:
        # Single atom case: there are no bonds
        avg_bond_dist = 0.0
    
    return np.array(comp_list + [vol_per_atom, density, avg_bond_dist], dtype=float)

print("Extracting composition + ASE structural features (Volume, Density, Bond Length)...")
X_all = np.vstack([improved_feat(a) for a in atoms_train_full])
X_test_final = np.vstack([improved_feat(a) for a in atoms_test_final])
y_all = df_train_full["hform"].values.astype(float)

# --- Train-Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# --- Preprocessing ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# --- Optimized GPR Model Setup ---
# WhiteKernel noise_level_bounds expanded to 1.0 based on your high RMSE outliers
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
         RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e2)) + \
         WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))

# --- Production Model Setup (using your winning parameters) ---
# We use the optimized values you just found as our starting point
optimized_kernel = ConstantKernel(1.13**2) * \
                   RBF(length_scale=7.14, length_scale_bounds=(1.0, 20.0)) + \
                   WhiteKernel(noise_level=0.139, noise_level_bounds=(1e-3, 0.5))

gpr = GaussianProcessRegressor(
    kernel=optimized_kernel, 
    normalize_y=True, 
    n_restarts_optimizer=1 
)

# --- Train and Evaluate ---
print(f"Training on {len(X_train)} samples...")
gpr.fit(X_train_scaled, y_train)

# Validation Predictions
y_pred_val = gpr.predict(X_val_scaled)

# --- RMSE and MAE Calculation ---
val_mae = mean_absolute_error(y_val, y_pred_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print("\n" + "="*30)
print("VALIDATION SET RESULTS")
print(f"Validation MAE:  {val_mae:.4f} eV/atom")
print(f"Validation RMSE: {val_rmse:.4f} eV/atom")
print(f"Optimized Kernel: {gpr.kernel_}")
print("="*30 + "\n")

# --- Final Prediction for Submission ---
y_pred_test = gpr.predict(X_test_scaled)
submission = pd.DataFrame({"id": df_test_final["id"].values, "hform": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

# --- Parity Plot ---
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred_val, alpha=0.4, color='blue', label='Validation Data')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual hform (eV/atom)')
plt.ylabel('Predicted hform (eV/atom)')
plt.title(f'Parity Plot (RMSE: {val_rmse:.3f})')
plt.legend()
plt.show()


#%%

from sklearn.model_selection import GridSearchCV

# --- Hyperparameter Testing (Grid Search) ---
print("Starting Hyperparameter Optimization...")

# Define the kernel with wide bounds
base_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# Define the model for searching
# Note: optimizer='fmin_l_bfgs_b' is used within each grid point
search_gpr = GaussianProcessRegressor(kernel=base_kernel, normalize_y=True)

# Define the grid of parameters to test
# We test different initial guesses for length_scale and noise
param_grid = {
    "kernel__k1__k2__length_scale": [0.1, 1.0, 10.0],
    "kernel__k2__noise_level": [0.01, 0.1, 0.5]
}

# Use KFold for cross-validation within the grid search
grid_search = GridSearchCV(
    search_gpr, 
    param_grid, 
    cv=3, 
    scoring='neg_root_mean_squared_error', 
    n_jobs=-1 # Uses all CPU cores
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

# Use the best model found
gpr = grid_search.best_estimator_



