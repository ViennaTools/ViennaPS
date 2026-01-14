import csv
import random
import os

def generate_random_rates_file(output_file, mode="3D", x_range=(0., 30.0), x_step=2.0,
                                y_range=(0., 30.0), y_step=2.0,
                                rate_min=1.5, rate_max=4.0):
    """
    Generates a random rate CSV file.

    Args:
        output_file (str): Path to output CSV.
        mode (str): "2D" or "3D".
        x_range (tuple): (x_min, x_max).
        x_step (float): Spacing in X.
        y_range (tuple): (y_min, y_max) for 3D mode.
        y_step (float): Spacing in Y for 3D mode.
        rate_min (float): Minimum random rate.
        rate_max (float): Maximum random rate.
    """
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        if mode == "2D":
            x_vals = [x_range[0] + i * x_step for i in range(int((x_range[1] - x_range[0]) / x_step) + 1)]

            for x in x_vals:
                rate = round(random.uniform(rate_min, rate_max), 5)
                writer.writerow([x, rate])

        elif mode == "3D":
            x_vals = [x_range[0] + i * x_step for i in range(int((x_range[1] - x_range[0]) / x_step) + 1)]
            y_vals = [y_range[0] + j * y_step for j in range(int((y_range[1] - y_range[0]) / y_step) + 1)]

            for y in y_vals:
                for x in x_vals:
                    rate = round(random.uniform(rate_min, rate_max), 5)
                    writer.writerow([x, y, rate])
        else:
            raise ValueError("Invalid mode. Choose '2D' or '3D'.")

    print(f"Generated {output_file} in {mode} mode with random rates.")

# -------------------------------
# Now generate multiple files!
# -------------------------------

# Parameters
mode = "3D"
n_files = 10 
output_folder = "rates"
os.makedirs(output_folder, exist_ok=True)

for i in range(n_files):
    output_file = os.path.join(output_folder, f"rates_{i}.csv")
    generate_random_rates_file(output_file, mode=mode)

print(f"Done. {n_files} files saved in '{output_folder}'")