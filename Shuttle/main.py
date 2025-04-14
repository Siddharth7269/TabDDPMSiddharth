#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import warnings

# For TSTR evaluation
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split # Not used in this specific TSTR setup
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # <-- Added Logistic Regression
import xgboost as xgb

# Optional: Suppress potential UserWarning from XGBoost regarding label encoding
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
# Optional: Suppress convergence warnings from Logistic Regression if needed
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

# -------------------------------
# Define Time Embedding Module
# -------------------------------
class SinusoidalPosEmb(nn.Module):
    """
    Creates sinusoidal positional embeddings for the timestep 't'.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # The dimension of the embedding vector

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): A 1D tensor of timesteps, shape (batch_size,)
        Returns:
            torch.Tensor: Positional embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        # Calculate embedding frequencies (denominator term)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Calculate embedding arguments (time * frequency)
        embeddings = time[:, None] * embeddings[None, :]
        # Apply sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Handle odd dimensions if necessary (though usually dim is even)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

# -------------------------------
# Define Residual Block Module
# -------------------------------
class ResidualBlock(nn.Module):
    """
    A residual block with Layer Normalization and SiLU activation.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU() # SiLU (Swish) often works well
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, dim)
        Returns:
            torch.Tensor: Output tensor, shape (batch_size, dim)
        """
        residual = x
        out = self.norm1(x)
        out = self.act1(out)
        out = self.linear1(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.linear2(out)
        # Add the residual connection
        out = out + residual
        return out

# -------------------------------
# Define the Complex MLP model for diffusion (TabDDPM)
# -------------------------------
class ComplexMLP(nn.Module):
    """
    A more complex MLP incorporating time embeddings and residual blocks
    to predict the noise added to data in a diffusion process.
    """
    def __init__(self, input_dim, hidden_dim, num_residual_blocks, time_emb_dim):
        """
        Args:
            input_dim (int): Dimension of the input data (number of features).
            hidden_dim (int): Hidden dimension used throughout the network.
            num_residual_blocks (int): Number of residual blocks to stack.
            time_emb_dim (int): Dimension for the sinusoidal time embedding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim

        # --- Time Embedding Layers ---
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        # MLP to project time embedding to the main hidden dimension
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- Main Network Layers ---
        # Initial projection of the input data x_t
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])

        # Final projection layer to predict noise (output dim = input dim)
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_t, t):
        """
        Forward pass through the network.

        Args:
            x_t (torch.Tensor): Noisy input data at timestep t, shape (batch_size, input_dim).
            t (torch.Tensor): Timesteps for each sample in the batch, shape (batch_size,).

        Returns:
            torch.Tensor: Predicted noise (epsilon), shape (batch_size, input_dim).
        """
        # 1. Compute time embedding and project it
        time_emb = self.time_embedding(t) # (batch_size, time_emb_dim)
        time_emb = self.time_mlp(time_emb) # (batch_size, hidden_dim)

        # 2. Project input data
        x_proj = self.input_proj(x_t) # (batch_size, hidden_dim)

        # 3. Combine input projection with time embedding
        h = x_proj + time_emb # Broadcasting adds time_emb

        # 4. Pass through residual blocks
        for block in self.residual_blocks:
            h = block(h) # (batch_size, hidden_dim)

        # 5. Final projection to output dimension
        output = self.final_proj(h) # (batch_size, input_dim)

        return output


# -------------------------------
# Diffusion schedule helper function
# -------------------------------
def get_alpha_bar(T, beta_start, beta_end, device):
    """
    Compute a linear beta schedule,
    alphas = 1 - betas,
    and alpha_bar = cumulative product of alphas.
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


# -------------------------------
# Training function for TabDDPM on the real dataset
# -------------------------------
def train_tabddpm(data_tensor, hparams, device):
    """
    Trains the TabDDPM (diffusion model using ComplexMLP) on the entire data_tensor.
    data_tensor should be of shape [num_samples, num_features].
    """
    dataset = TensorDataset(data_tensor)
    # Consider enabling pin_memory=True if using GPU and data fits in memory
    # Adjust num_workers based on your system
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True,
                            num_workers=min(4, os.cpu_count() if os.cpu_count() else 1),
                            pin_memory=torch.cuda.is_available())


    T = hparams.T  # total diffusion timesteps
    betas, alphas, alpha_bar = get_alpha_bar(T, hparams.beta_start, hparams.beta_end, device)

    input_dim = data_tensor.shape[1]
    # Instantiate the ComplexMLP model
    model = ComplexMLP(
        input_dim=input_dim,
        hidden_dim=hparams.hidden_dim,
        num_residual_blocks=hparams.num_residual_blocks,
        time_emb_dim=hparams.time_emb_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    num_epochs = hparams.epochs

    print(f"Starting TabDDPM training with ComplexMLP ({hparams.num_residual_blocks} blocks)...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train() # Set model to training mode
        for batch in dataloader:
            x0 = batch[0].to(device)  # clean data sample
            batch_size = x0.shape[0]

            # Sample random timesteps for each sample in batch
            t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long) # Ensure t is LongTensor

            # Sample Gaussian noise
            epsilon = torch.randn_like(x0)

            # Get corresponding alpha_bar for each sample
            alpha_bar_t = alpha_bar[t].view(batch_size, 1)

            # Forward diffusion: generate x_t = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*epsilon
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon

            # Predict noise using the model (pass x_t and t separately)
            predicted_epsilon = model(x_t, t)

            # Calculate loss
            loss = F.mse_loss(predicted_epsilon, epsilon)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping (can help stabilize training, especially with deeper networks)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / len(dataset) # Use len(dataset) for exact average
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs -1: # Print less frequently
             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    print("TabDDPM training complete!")
    # Return the trained model and the diffusion schedule components for sampling.
    return model, T, betas, alphas, alpha_bar


# -------------------------------
# Reverse sampling function for TabDDPM
# -------------------------------
@torch.inference_mode() # More efficient than torch.no_grad() for inference
def sample_tabddpm(model, T, betas, alphas, alpha_bar, num_samples, input_dim, device, batch_size=512):
    """
    Generates synthetic data by starting from pure noise and
    iteratively applying the reverse diffusion process using the ComplexMLP.
    Uses batching for potentially large num_samples.
    Returns a tensor of shape [num_samples, input_dim].
    """
    print(f"Starting sampling for {num_samples} samples using ComplexMLP...")
    model.eval() # Set model to evaluation mode

    all_samples = []
    num_generated = 0
    num_batches = math.ceil(num_samples / batch_size)

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - num_generated)
        print(f"  Generating batch {i+1}/{num_batches} (size {current_batch_size})...")

        # Start from pure Gaussian noise for the current batch
        x = torch.randn(current_batch_size, input_dim, device=device)

        # Iteratively denoise from timestep T-1 to 0.
        for t in reversed(range(T)):
            # Create a tensor of the current timestep t for the batch
            t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long) # Ensure t is LongTensor

            # Predict noise using the model
            predicted_epsilon = model(x, t_tensor)

            # Get schedule parameters for timestep t
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bar[t]
            beta_t = betas[t]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_t_inv = 1.0 / torch.sqrt(alpha_t)
            beta_t_over_sqrt_one_minus_alpha_bar_t = beta_t / sqrt_one_minus_alpha_bar_t

            # For all but the last step, add additional noise.
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

            # Reverse update rule (DDPM formula)
            x = sqrt_alpha_t_inv * (x - beta_t_over_sqrt_one_minus_alpha_bar_t * predicted_epsilon) \
                + torch.sqrt(beta_t) * z

            # Optional: print progress within a batch for long sampling
            # if t % 200 == 0 and i == 0: # Print only for first batch
            #      print(f"    Batch {i+1} - Timestep {t}/{T}")


        all_samples.append(x.cpu()) # Move generated batch to CPU
        num_generated += current_batch_size

    print("Sampling complete.")
    return torch.cat(all_samples, dim=0) # Concatenate all batches


# -------------------------------
# Function to load real dataset from file
# -------------------------------
def load_dataset(file_path):
    """
    Loads the dataset from file_path.
    Assumes the file is whitespace-delimited and numeric.
    Returns a numpy array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    print(f"Loading dataset from: {file_path}")
    try:
        # Try loading with space as delimiter first
        try:
             data = np.loadtxt(file_path)
        except ValueError:
             print("Whitespace delimiter failed, trying comma delimiter...")
             # If that fails, try comma delimiter
             data = np.loadtxt(file_path, delimiter=',')

        print(f"Dataset loaded successfully with shape: {data.shape}")
        if data.ndim == 1: # Handle case where loadtxt might return 1D array if file has only one line
             data = data.reshape(1, -1)
        # Check for NaNs or Infs which can cause issues later
        if np.isnan(data).any() or np.isinf(data).any():
            print("Warning: Dataset contains NaN or Inf values. Attempting to replace with 0.")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0, or consider other imputation
        return data
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        raise


# -------------------------------
# TSTR evaluation: Train on Synthetic, Test on Real (Verbose Version)
# -------------------------------
def tstr_evaluation_verbose(syn_data, real_data):
    """
    Performs TSTR (Train on Synthetic, Test on Real) evaluation with verbose output.
    MODIFIED: Maps rounded synthetic targets to the closest class found in real data
              to handle class mismatch errors during evaluation.

    Both syn_data and real_data are expected to be numpy arrays.
    Assumes the last column is the target variable.

    Trains multiple classifiers on synthetic data and evaluates them on real data.
    Prints detailed progress during training and final accuracy scores.

    Args:
        syn_data (np.ndarray): Synthetic dataset (features + target).
        real_data (np.ndarray): Real dataset (features + target).

    Returns:
        dict: A dictionary containing the accuracy scores and trained models for each classifier.
              Example: {'MLP': {'accuracy': 0.85, 'model': <MLPClassifier object>}, ...}
    """
    print("\n--- Starting Verbose TSTR Evaluation ---")

    # == Step 1: Data Preparation and Class Alignment ==
    print("\n[Step 1: Data Preparation & Class Alignment]")
    try:
        if syn_data is None or syn_data.shape[0] == 0:
             print("Error: Synthetic data is empty or None.")
             return None
        if real_data is None or real_data.shape[0] == 0:
             print("Error: Real data is empty or None.")
             return None
        if syn_data.shape[1] != real_data.shape[1] or syn_data.shape[1] < 2:
            raise ValueError(f"Incompatible data shapes or insufficient columns. "
                             f"Syn: {syn_data.shape}, Real: {real_data.shape}")

        # --- Real Data ---
        print("Processing real data...")
        X_real = real_data[:, :-1]
        # Get unique real classes FIRST, then cast y_real
        y_real_raw = real_data[:, -1]
        unique_real_classes = np.unique(y_real_raw.astype(int)) # Get the definitive integer classes from real data
        y_real = y_real_raw.astype(int) # Cast to int for classification
        print(f"  Real data shapes: X_real={X_real.shape}, y_real={y_real.shape}")
        print(f"  Unique real target classes found: {unique_real_classes}")
        if len(unique_real_classes) == 0:
            print("Error: No unique classes found in real data target.")
            return None

        # --- Synthetic Data ---
        print("Processing synthetic data and aligning classes...")
        X_syn = syn_data[:, :-1]
        # Round synthetic target first
        y_syn_rounded_raw = np.rint(syn_data[:, -1])
        print(f"  Unique synthetic target values (after rounding, before alignment): {np.unique(y_syn_rounded_raw.astype(int))}")

        # **Class Alignment Step:** Map rounded synthetic values to nearest real class
        print(f"  Aligning rounded synthetic targets to the nearest class in {unique_real_classes}...")
        # Define a helper function for mapping
        def map_to_nearest(value, valid_classes):
            return valid_classes[np.argmin(np.abs(valid_classes - value))]

        # Apply the mapping vectorized (more efficient) - requires numpy broadcasting
        # Calculate absolute differences between each syn value and all real classes
        diffs = np.abs(y_syn_rounded_raw[:, np.newaxis] - unique_real_classes)
        # Find the index of the minimum difference for each syn value
        nearest_indices = np.argmin(diffs, axis=1)
        # Use these indices to get the corresponding nearest real class
        y_syn_aligned = unique_real_classes[nearest_indices]

        # # Alternative (slower) list comprehension version:
        # y_syn_aligned = np.array([map_to_nearest(val, unique_real_classes) for val in y_syn_rounded_raw])

        y_syn = y_syn_aligned.astype(int) # Final synthetic target, cast to int

        print(f"  Synthetic data shapes after alignment: X_syn={X_syn.shape}, y_syn={y_syn.shape}")
        print(f"  Unique synthetic target classes (after alignment): {np.unique(y_syn)}")

        # Check if alignment resulted in valid classes
        if not np.all(np.isin(np.unique(y_syn), unique_real_classes)):
             print("Warning: Class alignment might have failed. Check synthetic data processing.")
        # Check if classes match now (they should, by definition of the alignment)
        if set(np.unique(y_syn)) != set(unique_real_classes):
            print("\nWarning: Unique classes in aligned synthetic and real target data still do not match perfectly (this shouldn't happen after alignment).")
            print(f"  Aligned Synthetic classes: {np.unique(y_syn)}")
            print(f"  Real classes:              {unique_real_classes}")


    except IndexError:
        print("Error: Could not split data. Ensure input arrays have at least two columns.")
        return None
    except Exception as e:
        print(f"An error occurred during data preparation/alignment: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None

    # == Step 2: Feature Scaling ==
    # (This part remains the same as before)
    print("\n[Step 2: Feature Scaling]")
    print("Initializing StandardScaler...")
    scaler = StandardScaler()
    print("Fitting StandardScaler on synthetic features (X_syn)...")
    X_syn_fit = X_syn if X_syn.ndim > 1 else X_syn.reshape(-1, 1)
    if np.any(np.std(X_syn_fit, axis=0) == 0):
        print("Warning: Some features in synthetic data have zero variance. Scaling might be affected.")
    scaler.fit(X_syn_fit)
    print("Transforming synthetic features (X_syn)...")
    X_syn_scaled = scaler.transform(X_syn_fit)
    print("Transforming real features (X_real) using the same scaler...")
    X_real_fit = X_real if X_real.ndim > 1 else X_real.reshape(-1, 1)
    X_real_scaled = scaler.transform(X_real_fit)
    print("Feature scaling complete.")

    print("\n***** TSTR Evaluation: Training on aligned synthetic, testing on real data *****")

    results = {}
    # Use the actual unique real classes for determining multiclass settings
    num_class = len(unique_real_classes)

    # == Model 1: MLP Classifier (Scikit-learn) ==
    print("\n===== [MLP Classifier Evaluation] =====")
    print("Initializing MLPClassifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=300, random_state=42,
        verbose=False, # Reduced verbosity for cleaner logs, set True if needed
        early_stopping=True, n_iter_no_change=15
    )
    print(f"\nStarting MLP training on scaled aligned synthetic data (X_syn_scaled, y_syn)...")
    try:
        mlp.fit(X_syn_scaled, y_syn)
        print("\nMLP training finished.")
        # Optional: print convergence info if needed
        # print(f"  MLP converged after {mlp.n_iter_} iterations.")
        # print(f"  Final MLP training loss: {mlp.loss_:.6f}")

        print("Evaluating MLP on scaled real data (X_real_scaled)...")
        y_pred_mlp = mlp.predict(X_real_scaled)
        mlp_acc = accuracy_score(y_real, y_pred_mlp)
        print(f"--> MLP Classifier Accuracy on Real Data: {mlp_acc:.4f}")
        results["MLP"] = {"accuracy": mlp_acc, "model": mlp}
    except Exception as e:
        print(f"An error occurred during MLP training or evaluation: {e}")
        results["MLP"] = {"accuracy": None, "model": None, "error": str(e)}
    print("----------------------------------------")


    # == Model 2: XGBoost Classifier ==
    print("\n===== [XGBoost Classifier Evaluation] =====")
    print("Initializing XGBoost Classifier...")
    eval_metric = 'mlogloss' if num_class > 2 else 'logloss'
    objective = 'multi:softprob' if num_class > 2 else 'binary:logistic'
    print(f"  Using objective: '{objective}', eval_metric: '{eval_metric}' for {num_class} classes.")

    # Map potentially non-contiguous labels (like -4, -3, 0, 1, 5...) to contiguous 0, 1, 2... for XGBoost
    # Create mapping from original label to 0-based index
    label_map = {label: i for i, label in enumerate(unique_real_classes)}
    y_syn_mapped = np.array([label_map[label] for label in y_syn])
    y_real_mapped = np.array([label_map[label] for label in y_real])
    print(f"  Mapped target labels for XGBoost training/evaluation to range 0..{num_class-1}")

    xgb_clf = xgb.XGBClassifier(
        objective=objective,
        num_class=num_class if num_class > 2 else None, # Only needed for multi:softprob
        use_label_encoder=False,
        eval_metric=eval_metric,
        random_state=42,
        n_estimators=200,
        early_stopping_rounds=20
    )
    print(f"\nStarting XGBoost training on scaled aligned synthetic data (X_syn_scaled, y_syn_mapped)...")
    # Use mapped real data for evaluation set
    eval_set = [(X_syn_scaled, y_syn_mapped), (X_real_scaled, y_real_mapped)]
    try:
        xgb_clf.fit(
            X_syn_scaled,
            y_syn_mapped, # Train with 0-based labels
            eval_set=eval_set,
            verbose=False # Set to 20 for periodic output
        )
        print("\nXGBoost training finished.")
        # if hasattr(xgb_clf, 'best_iteration'): print(f"  Best iteration: {xgb_clf.best_iteration}")

        print("Evaluating XGBoost on scaled real data (X_real_scaled)...")
        y_pred_xgb_mapped = xgb_clf.predict(X_real_scaled)
        # IMPORTANT: Accuracy score needs the ORIGINAL labels, not the 0-based ones
        xgb_acc = accuracy_score(y_real, y_pred_xgb_mapped) # Compare original y_real with 0-based prediction
        # ^^^ Correction: Need to compare y_real_mapped with y_pred_xgb_mapped for accuracy
        xgb_acc = accuracy_score(y_real_mapped, y_pred_xgb_mapped)

        print(f"--> XGBoost Classifier Accuracy on Real Data: {xgb_acc:.4f}")
        results["XGBoost"] = {"accuracy": xgb_acc, "model": xgb_clf} # Note: model predicts 0-based labels now
    except Exception as e:
        print(f"An error occurred during XGBoost training or evaluation: {e}")
        results["XGBoost"] = {"accuracy": None, "model": None, "error": str(e)}
    print("----------------------------------------")


    # == Model 3: Random Forest Classifier ==
    print("\n===== [Random Forest Classifier Evaluation] =====")
    print("Initializing Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=150, random_state=42, verbose=0, n_jobs=-1,
        max_depth=20, min_samples_leaf=3
    )
    print(f"\nStarting Random Forest training on scaled aligned synthetic data (X_syn_scaled, y_syn)...")
    try:
        rf.fit(X_syn_scaled, y_syn) # RF handles non-contiguous labels directly
        print("\nRandom Forest training finished.")
        print("Evaluating Random Forest on scaled real data (X_real_scaled)...")
        y_pred_rf = rf.predict(X_real_scaled)
        rf_acc = accuracy_score(y_real, y_pred_rf)
        print(f"--> Random Forest Classifier Accuracy on Real Data: {rf_acc:.4f}")
        results["RandomForest"] = {"accuracy": rf_acc, "model": rf}
    except Exception as e:
        print(f"An error occurred during Random Forest training or evaluation: {e}")
        results["RandomForest"] = {"accuracy": None, "model": None, "error": str(e)}
    print("----------------------------------------")


    # == Model 4: Logistic Regression ==
    print("\n===== [Logistic Regression Evaluation] =====")
    print("Initializing Logistic Regression...")
    log_reg = LogisticRegression(
        solver='liblinear', random_state=42, max_iter=1000, C=1.0, penalty='l2'
        # multi_class='auto' # Handles label types automatically usually
    )
    print(f"\nStarting Logistic Regression training on scaled aligned synthetic data (X_syn_scaled, y_syn)...")
    try:
        log_reg.fit(X_syn_scaled, y_syn) # LogReg handles non-contiguous labels directly
        print("\nLogistic Regression training finished.")
        print("Evaluating Logistic Regression on scaled real data (X_real_scaled)...")
        y_pred_log_reg = log_reg.predict(X_real_scaled)
        log_reg_acc = accuracy_score(y_real, y_pred_log_reg)
        print(f"--> Logistic Regression Accuracy on Real Data: {log_reg_acc:.4f}")
        results["LogisticRegression"] = {"accuracy": log_reg_acc, "model": log_reg}
    except Exception as e:
        print(f"An error occurred during Logistic Regression training or evaluation: {e}")
        results["LogisticRegression"] = {"accuracy": None, "model": None, "error": str(e)}
    print("----------------------------------------")


    print("\n***** End of Verbose TSTR Evaluation *****\n")
    return results


# -------------------------------
# Hyperparameter parsing and script entry point
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TabDDPM Training with ComplexMLP and TSTR Evaluation")

    # --- Diffusion hyperparameters ---
    parser.add_argument("--T", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting beta value for noise schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending beta value for noise schedule")

    # --- Dataset hyperparameters ---
    # Made data_file required
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to the real dataset file (whitespace or comma-delimited, numeric)")

    # --- Training hyperparameters for TabDDPM ---
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs for TabDDPM") # Increased default
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for TabDDPM training") # Increased default
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for Adam optimizer") # Adjusted default

    # --- Network architecture hyperparameters (ComplexMLP) ---
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size for the MLP")
    parser.add_argument("--num_residual_blocks", type=int, default=4, help="Number of residual blocks in the MLP")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Dimension for the sinusoidal time embedding")

    # --- Sampling hyperparameter ---
    parser.add_argument("--num_syn_samples", type=int, default=None,
                        help="Number of synthetic samples to generate (defaults to size of real dataset)")
    parser.add_argument("--output_file", type=str, default="synthetic_data.csv",
                        help="Filename to save the generated synthetic data")
    # Argument to control TSTR evaluation
    parser.add_argument("--skip_tstr", action='store_true', help="Skip the TSTR evaluation step")


    return parser.parse_args()


def main():
    hparams = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================")
    print(f"        TabDDPM & TSTR Evaluation Script")
    print(f"=============================================")
    print(f"Using device: {device}")
    print("\nHyperparameters:")
    for key, value in vars(hparams).items():
        print(f"  {key}: {value}")
    print("---------------------------------------------")


    # Load the real dataset strictly from the provided file path.
    try:
        data_np = load_dataset(hparams.data_file)
    except FileNotFoundError:
        print(f"Error: Dataset file '{hparams.data_file}' not found. Exiting.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}. Exiting.")
        return

    # Basic check after loading
    if data_np is None or data_np.size == 0:
        print("Error: Loaded dataset is empty or None. Exiting.")
        return
    if data_np.ndim != 2 or data_np.shape[1] < 2:
         print(f"Error: Dataset must have at least 2 columns (features + target). Shape is {data_np.shape}. Exiting.")
         return


    # Determine number of synthetic samples
    num_real_samples, input_dim = data_np.shape
    num_syn_samples = hparams.num_syn_samples if hparams.num_syn_samples is not None else num_real_samples
    print(f"\nReal dataset shape: {data_np.shape}")
    print(f"Will generate {num_syn_samples} synthetic samples.")
    print("---------------------------------------------")


    # Convert real data to a PyTorch tensor of type float.
    # Load to CPU first, then move within training if needed
    data_tensor = torch.tensor(data_np, dtype=torch.float, device='cpu')

    # Train the diffusion model (TabDDPM) on the real dataset.
    print("\nStarting Diffusion Model Training...")
    model, T, betas, alphas, alpha_bar = train_tabddpm(data_tensor, hparams, device)
    print("---------------------------------------------")


    # Generate synthetic data using the trained model.
    print("\nStarting Synthetic Data Generation...")
    # Use a reasonable batch size for sampling if generating many samples
    sampling_batch_size = min(hparams.batch_size * 2, 1024) # Example batch size heuristic
    synthetic_tensor = sample_tabddpm(model, T, betas, alphas, alpha_bar,
                                      num_syn_samples, input_dim, device,
                                      batch_size=sampling_batch_size)
    synthetic_data = synthetic_tensor.cpu().numpy()

    # Post-process synthetic data: round the target (last column)
    # Ensure rounding happens correctly even if target is not integer-like originally
    print("Post-processing synthetic data (rounding target column)...")
    if synthetic_data.shape[1] > 0:
         # Round first, then we will cast to int inside tstr_evaluation
         synthetic_data[:, -1] = np.rint(synthetic_data[:, -1])
         print(f"  Unique target values in generated data after rounding: {np.unique(synthetic_data[:, -1])}")
    else:
         print("Warning: Synthetic data has no columns. Skipping target rounding.")


    # Save synthetic data to a CSV file.
    try:
        # Save with more precision for features, format target as int if desired after rounding
        # Using a flexible format string
        np.savetxt(hparams.output_file, synthetic_data, delimiter=",", fmt="%.6g")
        print(f"Synthetic dataset saved to {hparams.output_file}")
    except Exception as e:
        print(f"Error saving synthetic data to {hparams.output_file}: {e}")
    print("---------------------------------------------")


    # TSTR Evaluation: train models on synthetic data, test on real data.
    if not hparams.skip_tstr:
         if synthetic_data.shape[0] > 0 and data_np.shape[0] > 0:
             tstr_results = tstr_evaluation_verbose(synthetic_data, data_np)

             if tstr_results:
                print("\nFinal TSTR Accuracies Summary:")
                print("---------------------------------------------")
                for model_name, result in tstr_results.items():
                    acc = result.get('accuracy', None)
                    if acc is not None:
                         print(f"  {model_name}: {acc:.4f}")
                    else:
                         error_msg = result.get('error', 'Unknown error')
                         print(f"  {model_name}: Error during evaluation ({error_msg})")
                print("---------------------------------------------")
             else:
                 print("TSTR evaluation could not be completed due to errors during setup.")
         else:
            print("Skipping TSTR evaluation because either synthetic or real data is empty.")
    else:
        print("Skipping TSTR evaluation as requested by --skip_tstr flag.")

    print("\nScript finished.")
    print(f"=============================================")


if __name__ == "__main__":
    main()