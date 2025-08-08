import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import os

# Define the search space for PID parameters
space  = [
    Real(0.0, 10.0, name='Kp'),
    Real(0.0, 5.0, name='Ki'),
    Real(0.0, 1.0, name='Kd')
]

def calculate_settling_time(flow, setpoint, tolerance=0.02):
    """
    Calculate settling time: time until flow stays within tolerance band around setpoint.
    """
    steady_band_low = setpoint * (1 - tolerance)
    steady_band_high = setpoint * (1 + tolerance)

    settled_indices = np.where((flow >= steady_band_low) & (flow <= steady_band_high))[0]

    if len(settled_indices) == 0:
        # Never settled
        return float('inf')
    
    # Settling time = time at first index where it stays within tolerance till the end
    for idx in settled_indices:
        # Check if flow remains within tolerance for rest of the samples
        if np.all((flow[idx:] >= steady_band_low) & (flow[idx:] <= steady_band_high)):
            return idx * 0.1  # Assuming 0.1 s between samples (100 ms)
    return float('inf')

def cost_from_logfile(filepath):
    """
    Load flow log CSV and compute cost (settling time + overshoot penalty).
    """
    if not os.path.exists(filepath):
        print(f"Log file {filepath} not found, assigning high cost.")
        return 1e6  # Large penalty if log not found

    data = pd.read_csv(filepath)
    flow = data['flow_value'].values
    setpoint = data['setpoint'].values

    # Calculate settling time
    settling_time = calculate_settling_time(flow, setpoint[-1])

    # Calculate overshoot
    overshoot = np.max(flow) - setpoint[-1]
    overshoot = max(overshoot, 0)  # Only positive overshoot counts

    # Cost function: weight settling time and overshoot
    cost = settling_time + 10 * overshoot
    print(f"Settling time: {settling_time:.2f}s, Overshoot: {overshoot:.2f}, Cost: {cost:.2f}")
    return cost

@use_named_args(space)
def cost_function(Kp, Ki, Kd):
    # Format filename based on PID parameters rounded to 3 decimals
    filename = f"log_Kp_{Kp:.3f}_Ki_{Ki:.3f}_Kd_{Kd:.3f}.csv"
    
    print(f"\nEvaluating PID: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
    
    # In your real use case, at this point:
    # 1. You run your C# system with these PID parameters.
    # 2. You save the flow log as the above filename.
    # 3. You then run this script or continue to optimize using that logfile.

    # For demonstration, we calculate cost from existing log file:
    cost = cost_from_logfile(filename)
    
    return cost

def main():
    print("Starting Bayesian Optimization for PID tuning...\n")
    result = gp_minimize(
        func=cost_function,
        dimensions=space,
        acq_func='EI',          # Expected Improvement
        n_calls=20,             # Number of iterations (runs)
        n_random_starts=5,      # Initial random sampling
        random_state=42
    )

    print("\nOptimization finished!")
    print(f"Best PID parameters found: Kp={result.x[0]:.3f}, Ki={result.x[1]:.3f}, Kd={result.x[2]:.3f}")
    print(f"Minimum cost: {result.fun:.3f}")

if __name__ == "__main__":
    main()
