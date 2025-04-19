import csv
import random

mode = "3D" 

output_file = f"rates{mode}.csv"

# Grid setup
x_range = (0.0, 30.0)
x_step = 2.0

# For 3D
y_range = (0.0, 30.0)
y_step = 2.0

# Rate range
rate_min = 0.01
rate_max = 0.08

# Generate CSV
with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)

    if mode == "2D":
        writer.writerow(["x", "rate"])
        x_vals = [x_range[0] + i * x_step for i in range(int((x_range[1] - x_range[0]) / x_step) + 1)]

        for x in x_vals:
            rate = round(random.uniform(rate_min, rate_max), 5)
            writer.writerow([x, rate])

    elif mode == "3D":
        writer.writerow(["x", "y", "rate"])
        x_vals = [x_range[0] + i * x_step for i in range(int((x_range[1] - x_range[0]) / x_step) + 1)]
        y_vals = [y_range[0] + j * y_step for j in range(int((y_range[1] - y_range[0]) / y_step) + 1)]

        for y in y_vals:
            for x in x_vals:
                rate = round(random.uniform(rate_min, rate_max), 5)
                writer.writerow([x, y, rate])
    else:
        raise ValueError("Invalid mode. Choose '2D' or '3D'.")

print(f"Generated {output_file} in {mode} mode with random rates.")
