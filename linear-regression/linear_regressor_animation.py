import time
import matplotlib.pyplot as plt

# 1. Dataset Setup
X = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# 2. Loss Function
def train_lr(X, y, w, b): 
    predictions = [(w * x) + b for x in X]
    mse = [(y_t - y_p)**2 for y_t, y_p in zip(y, predictions)]
    return sum(mse) / len(y)

# 3. Hyperparameters
w, b = -3.0, 0.0  # Initial weight and bias
alpha = 0.05      # Learning rate
h = 0.00001       # Step size for numerical gradient
epochs = 60       

# 4. Tracking Lists for Convergence Visuals
loss_history = []
epoch_history = []

# 5. Figure Layout Setup (1 Row, 2 Columns)
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Linear Regression Optimization Journey", fontsize=14, fontweight='bold')

# --- STEP 1: INITIAL STABLE WINDOW LOAD ---
# Plot initial baseline before gradient descent begins
ax1.scatter(X, y, color='crimson', s=100, zorder=3, label='Actual Data')
initial_preds = [(w * x) + b for x in X]
ax1.plot(X, initial_preds, color='dodgerblue', alpha=0.5, linestyle='--', label='Initial Guess')
ax1.set_xlim(0, 6)
ax1.set_ylim(-5, 15)
ax1.set_xlabel("Hours Studied (X)")
ax1.set_ylabel("Marks (y)")
ax1.set_title("Line Fitting Status")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.4)

# Setup empty Convergence plot canvas
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, 200) # Big initial loss boundary
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_title("Convergence Curve (Error Reduction)")
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
print("Window loaded. Animation starts in 2 seconds...")
plt.pause(2.0)  # Holds the empty window on screen so the user can see it load

# --- STEP 2: ANIMATION & TRAINING LOOP ---
for epoch in range(epochs):
    current_mse = train_lr(X, y, w, b)
    
    # Store metrics for convergence tracking
    loss_history.append(current_mse)
    epoch_history.append(epoch)
    
    # Gradient Calculations
    slope_w = (train_lr(X, y, w + h, b) - current_mse) / h
    slope_b = (train_lr(X, y, w, b + h) - current_mse) / h
    
    # Parameter Updates
    w -= alpha * slope_w
    b -= alpha * slope_b
    
    current_predictions = [(w * x) + b for x in X]
    
    # --- UPDATE PLOT 1: Regression Line ---
    ax1.clear()
    ax1.scatter(X, y, color='crimson', s=100, zorder=3, label='Actual Data')
    ax1.plot(X, current_predictions, color='dodgerblue', linewidth=2.5, 
             label=f'Model: y = {w:.2f}x + {b:.2f}')
    ax1.set_xlim(0, 6)
    ax1.set_ylim(-5, 15)
    ax1.set_xlabel("Hours Studied (X)")
    ax1.set_ylabel("Marks (y)")
    ax1.set_title(f"Fit Status | Epoch: {epoch}")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # --- UPDATE PLOT 2: Convergence Curve ---
    ax2.clear()
    ax2.plot(epoch_history, loss_history, color='purple', linewidth=2, label='Current Loss')
    ax2.scatter(epoch, current_mse, color='black', s=40, zorder=4) # Flashing head tracker
    ax2.set_xlim(0, epochs)
    # Dynamically scale y-axis limit so you can watch tiny micro-adjustments at the end
    ax2.set_ylim(-5, max(loss_history) * 1.1) 
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title(f"Convergence | Loss: {current_mse:.4f}")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # Render Frame smoothly
    plt.pause(0.08)

# Turn off interactive tracking & lock final window open
plt.ioff()
print(f"\nConvergence Complete! Final Line Equation: y = {w:.2f}x + {b:.2f}")
plt.show()
