##############################################################
# TecPlotReader.py
# A GUI and batch exporter for Tecplot particle distribution data.
# It allows users to visualize and save particle distributions
# from Tecplot files, with options for trimming and selecting variables.
# Usage:
#   python TecPlotReader.py -f data.pdt
#   python TecPlotReader.py -f data.pdt -i variable_name -o output.txt
#   python TecPlotReader.py -f data.pdt -i variable_name --fit
#   python TecPlotReader.py  # to open the GUI
# Requires: tkinter, matplotlib, numpy, scipy
#####################################################################

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import re
import argparse
import sys
from scipy.optimize import curve_fit
from scipy.stats import vonmises
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


# --- Data Handling Functions ---
def read_tecplot_data(filename, thetaKey, energyKey, zone_lines_max=2):
    datafile = open(filename, "r")
    mode = -1  # 0=vars,1=zone,2=data
    zone_lines_read = 0
    variables = []
    data_dict = {}
    tmp_data = []
    var_counter = 0
    data_i = data_j = 0
    I = J = 0

    for line in datafile:
        if not line.strip():
            continue
        upper = line.upper()
        if "VARIABLES" in upper:
            mode = 0
        elif "ZONE" in upper:
            mode = 1

        if mode == 0:
            for var in re.findall(r'"([^\"]+)"', line):
                # if var not in (thetaKey, energyKey):
                #     var = var.split()[0]  # take only the first part
                variables.append(var)
                data_dict[var] = []
        elif mode == 1:
            if zone_lines_read == 0:
                mI = re.search(r"I=\s*(\d+)", line)
                mJ = re.search(r"J=\s*(\d+)", line)
                if mI and mJ:
                    I, J = int(mI.group(1)), int(mJ.group(1))
            zone_lines_read += 1
            if zone_lines_read == zone_lines_max:
                mode = 2  # switch to data mode
        elif mode == 2:
            var = variables[var_counter]
            parts = line.split()
            tmp_data.extend(map(float, parts))
            data_i += len(parts)
            if data_i == I:
                data_i = 0
                data_dict[var].append(tmp_data)
                tmp_data = []
                data_j += 1
                if data_j == J:
                    data_j = 0
                    var_counter += 1
    datafile.close()

    # convert to arrays
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])

    if thetaKey not in data_dict or energyKey not in data_dict:
        raise ValueError(
            f"Required keys '{thetaKey}' or '{energyKey}' not found in the data."
        )

    # extract 1D theta and energy
    data_dict[thetaKey] = data_dict[thetaKey][0, :]
    data_dict[energyKey] = data_dict[energyKey][:, 0]
    return data_dict, variables


def trim_data(data, energy, eps=1e-4):
    low = next((i for i in range(data.shape[0]) if np.any(data[i] >= eps)), 0)
    high = data.shape[0]
    for i in range(data.shape[0]):
        if np.all(data[i] < eps) and all(
            np.all(data[j] < eps) for j in range(i, data.shape[0])
        ):
            high = i
            break
    return data[low:high, :], energy[low:high]


def plot_data(data_dict, key, thetaKey, energyKey, eps):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    xx, yy = np.meshgrid(data_dict[thetaKey], e)
    plt.figure()
    plt.pcolormesh(xx, yy, d)
    plt.colorbar(label="Distribution")
    plt.xlabel(thetaKey)
    plt.ylabel(energyKey)
    plt.title(f"Distribution: {key}")
    plt.show()


def line_plot(data_dict, key, thetaKey, energyKey, eps):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    theta = data_dict[thetaKey]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(e)):
        ax.plot(
            ys=theta,
            zs=d[i, :],
            xs=e[i],
            color=plt.cm.viridis(i / len(e)),
        )
    ax.set_ylabel(thetaKey)
    ax.set_zlabel(f"Distribution: {key}")
    ax.set_xlabel(energyKey)
    plt.tight_layout()
    plt.show()


def fit_data(
    data_dict,
    key,
    thetaKey,
    energyKey,
    eps,
    angle_p_cutoff=10000,
    useSin=False,
    verbose=False,
    vm=False,
    alpha=0.0,
    beta=2.0,
    n_gauss=None,
):
    def _powerCos(t, power):
        f = np.cos(t) ** power
        if useSin:
            f = np.abs(np.sin(t) * f)
        return f / np.max(f)

    def _vonMises(t, kappa):
        f = vonmises.pdf(t, kappa=kappa, loc=0, scale=np.pi / 2)
        if useSin:
            f = np.abs(np.sin(t) * f)
        return f / np.max(f)

    def _sum_of_gaussians(x, *params):
        """Sum of N zero-baseline Gaussians."""
        y = np.zeros_like(x)
        n = len(params) // 3
        for i in range(n):
            amp, mu, sigma = params[3 * i : 3 * i + 3]
            y += amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        return y

    def _fit_n_gaussians(energy, pdf, prominence=None, distance=None, num_gauss=None):
        offset = pdf.min()
        y = pdf - offset
        span = energy.max() - energy.min()

        # detect peaks for initial centers
        peaks, _ = find_peaks(
            y, prominence=prominence or (y.max() * 0.1), distance=distance or 2
        )
        peak_heights = y[peaks]
        peak_mus = energy[peaks]

        # pick up to num_gauss strongest peaks
        if num_gauss is None:
            num_gauss = len(peaks)
        if len(peaks) >= num_gauss:
            idx = np.argsort(peak_heights)[-num_gauss:]
            init_mus = list(peak_mus[idx])
            init_amps = list(peak_heights[idx])
        else:
            # use all found peaks
            init_mus = list(peak_mus)
            init_amps = list(peak_heights)
            # fill missing with evenly spaced guesses
            needed = num_gauss - len(init_mus)
            grid_mus = np.linspace(
                energy.min() + span / (2 * num_gauss),
                energy.max() - span / (2 * num_gauss),
                num_gauss,
            )
            for gm in grid_mus:
                if len(init_mus) >= num_gauss:
                    break
                # avoid duplicates
                if all(abs(gm - m0) > span / (2 * num_gauss) for m0 in init_mus):
                    init_mus.append(gm)
                    init_amps.append(y.max() * 0.5)
            # if still short, pad at center
            while len(init_mus) < num_gauss:
                init_mus.append(energy.mean())
                init_amps.append(y.max() * 0.5)

        # common sigma guess
        init_sigmas = [span / (4 * num_gauss)] * num_gauss

        # build p0 = [amp1, mu1, sigma1, amp2, mu2, sigma2, ...]
        p0 = []
        for A, M, S in zip(init_amps, init_mus, init_sigmas):
            p0 += [A, M, S]

        # fit
        popt, _ = curve_fit(_sum_of_gaussians, energy, y, p0=p0)

        # parse results
        fits = []
        for i in range(num_gauss):
            amp = popt[3 * i]
            mu = popt[3 * i + 1]
            sigma = popt[3 * i + 2]
            fits.append({"amp": amp, "mu": mu, "sigma": sigma})

        return fits, popt, offset

    angle_fit = _vonMises if vm else _powerCos
    angle_param = "Kappa" if vm else "Power"

    def dist_fit(t, e, p, gauss_p, offset):
        return angle_fit(t, p) * (_sum_of_gaussians(e, *gauss_p) + offset)

    dist, energy = trim_data(data_dict[key], data_dict[energyKey], eps)
    theta = data_dict[thetaKey]
    theta_fit = np.linspace(np.min(theta), np.max(theta), 1000)
    energy_fit = np.linspace(np.min(energy), np.max(energy), 1000)

    # fit angle distribution
    angle_p = np.zeros(len(energy))
    for i in range(len(energy)):
        sample = dist[i, :] * (
            beta - (beta - 1) * np.exp(-alpha * np.abs(np.deg2rad(theta)))
        )
        sample_max = np.max(sample)
        if sample_max > 0:
            popt, _ = curve_fit(
                angle_fit,
                np.deg2rad(theta),
                sample / sample_max,
                bounds=([1], [angle_p_cutoff]),
                p0=[100],
            )
            angle_p[i] = popt[0]
        else:
            angle_p[i] = angle_p_cutoff

    if verbose:
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3)  # leave room at bottom
        [line_dist] = ax.plot(theta, dist[0, :], label="Distribution")
        [line_fit] = ax.plot(
            theta_fit,
            angle_fit(np.deg2rad(theta_fit), angle_p[0]) * np.max(dist[0, :]),
            label="Fit",
        )
        ax.set_xlabel(thetaKey)
        ax.set_title(f"Energy = {energy[0]:.2f} eV, {angle_param} = {angle_p[0]:.2f}")

        # TextBox for the integer, centered
        axbox = plt.axes([0.4, 0.15, 0.2, 0.05])
        text_box = TextBox(axbox, "", initial=str(0))

        # “–” button
        axdec = plt.axes([0.33, 0.15, 0.05, 0.05])
        btn_dec = Button(axdec, "-")

        # “+” button
        axinc = plt.axes([0.62, 0.15, 0.05, 0.05])
        btn_inc = Button(axinc, "+")

        def update_plot():
            try:
                f = int(text_box.text)
                if f < 0 or f >= len(energy):
                    raise ValueError("Index out of bounds")
            except ValueError:
                return
            line_dist.set_ydata(dist[f, :])
            line_fit.set_ydata(
                angle_fit(np.deg2rad(theta_fit), angle_p[f]) * np.max(dist[f, :])
            )
            ax.set_title(
                f"Energy = {energy[f]:.2f} eV, {angle_param} = {angle_p[f]:.2f}"
            )
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        def on_inc(event):
            v = min(int(text_box.text) + 1, len(energy) - 1)
            text_box.set_val(str(v))
            update_plot()

        def on_dec(event):
            v = max(int(text_box.text) - 1, 0)
            text_box.set_val(str(v))
            update_plot()

        # wire up callbacks
        btn_inc.on_clicked(on_inc)
        btn_dec.on_clicked(on_dec)
        text_box.on_submit(lambda text: update_plot())

    # fit resulting parameter distribution
    angle_p_cut = angle_p < (angle_p_cutoff - 1)
    popt_p, _ = curve_fit(
        lambda x, a, b: a * x + b,
        energy[angle_p_cut],
        angle_p[angle_p_cut],
        bounds=([0, 0], [10, 10000]),
        p0=[1, 1],
    )

    if verbose:
        plt.figure()
        plt.plot(energy, angle_p, "-x", label=angle_param)
        plt.plot(
            energy[angle_p_cut],
            popt_p[0] * energy[angle_p_cut] + popt_p[1],
            "o--",
            label=f"{angle_param} Fit",
        )
        plt.axhline(angle_p_cutoff, color="red", linestyle="--", label="Cutoff")
        plt.xlabel(energyKey)
        plt.ylabel(angle_param)
        plt.title(f"{angle_param} Fit: a={popt_p[0]:.2f}, b={popt_p[1]:.2f}")
        plt.legend()

    # fit energy distribution
    dist_t = np.sum(dist, axis=1)
    t_max = np.max(dist_t)
    if t_max > 0:
        fits, popt_g, offset = _fit_n_gaussians(energy, dist_t, num_gauss=n_gauss)
        for i, fit in enumerate(fits):
            print(
                f"Gaussian {i}: amp={fit['amp']:.4f}, mu={fit['mu']:.2f}, sigma={fit['sigma']:.2f}"
            )
    else:
        return -1

    if verbose:
        plt.figure()
        plt.plot(energy, dist_t, label="Cumulative Distribution")
        plt.plot(
            energy_fit,
            _sum_of_gaussians(energy_fit, *popt_g) + offset,
            label="Gaussian Fit",
        )
        plt.xlabel(energyKey)
        plt.ylabel("Cumulative Distribution")
        plt.legend()

    tt, ee = np.meshgrid(theta_fit, energy_fit)
    d = dist_fit(np.deg2rad(tt), ee, popt_p[1], popt_g, offset)
    plt.figure()
    plt.pcolormesh(tt, ee, d, shading="auto")
    plt.colorbar(label="Distribution")
    plt.xlabel(thetaKey)
    plt.ylabel(energyKey)
    plt.title(f"{angle_param} {popt_p[1]:.2f}")
    plt.tight_layout()
    plt.show()

    return 0


def smooth_and_downsample(pdf2d, energies, angles, sigma=(1, 1), new_shape=(50, 50)):
    """
    Smooth a 2D PDF and downsample its grid.

    Args:
      pdf2d      : 2D array, shape (n_energy, n_angle).
      energies   : 1D array of length n_energy.
      angles     : 1D array of length n_angle.
      sigma      : tuple (sigma_energy, sigma_angle) for Gaussian smoothing.
      new_shape  : tuple (n_new_energy, n_new_angle).

    Returns:
      new_pdf        : 2D array, shape new_shape.
      new_energies   : 1D array, length n_new_energy.
      new_angles     : 1D array, length n_new_angle.
    """
    # 1) Smooth with a Gaussian filter
    smooth = gaussian_filter(pdf2d, sigma=sigma)

    # 2) Pick new indices by linear spacing
    n_e, n_a = pdf2d.shape
    n_e2, n_a2 = new_shape
    idx_e = np.linspace(0, n_e - 1, n_e2).astype(int)
    idx_a = np.linspace(0, n_a - 1, n_a2).astype(int)

    # 3) Downsample both PDF and axes
    new_pdf = smooth[np.ix_(idx_e, idx_a)]
    new_energies = energies[idx_e]
    new_angles = angles[idx_a]

    return new_pdf, new_energies, new_angles


def save_data(data_dict, key, thetaKey, energyKey, eps, outname):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    with open(outname, "w") as f:
        f.write(f"# {key}\n")
        f.write(" ".join(map(str, e)) + "\n")
        f.write(" ".join(map(str, data_dict[thetaKey])) + "\n")
        for row in d:
            f.write(" ".join(map(str, row)) + "\n")


def save_dist(dist, theta, energy, outname, key="Particle Distribution"):
    with open(outname, "w") as f:
        f.write(f"# {key}\n")
        f.write(" ".join(map(str, energy)) + "\n")
        f.write(" ".join(map(str, theta)) + "\n")
        for row in dist:
            f.write(" ".join(map(str, row)) + "\n")


# --- GUI ---
class TecplotGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        parent.title("Tecplot Distribution Viewer")
        parent.minsize(400, 300)
        self.data = {}
        self.vars = []

        # use a built‑in theme
        # style = ttk.Style()
        # style.theme_use("clam")

        # Options Input
        self.tk_theta = "THETA (DEG)"
        self.tk_energy = "ENERGY (EV)"
        self.entry_theta = tk.StringVar(value=self.tk_theta)
        self.entry_energy = tk.StringVar(value=self.tk_energy)
        self.entry_zone_lines = tk.IntVar(value=2)
        self.line_plot_var = tk.BooleanVar(value=False)
        self.fit_angle_p_cutoff = tk.IntVar(value=10000)
        self.fit_use_sin = tk.BooleanVar(value=False)
        self.fit_verbose = tk.BooleanVar(value=False)
        self.fit_function = tk.StringVar(value="powerCosine")
        self.fit_alpha = tk.DoubleVar(value=0.0)
        self.fit_beta = tk.DoubleVar(value=2.0)
        self.fit_n_gauss = tk.IntVar(value=0)
        self.path_fn = ""

        # File input
        frame_file = ttk.Frame(self)
        ttk.Label(frame_file, text="Filename:").pack(anchor="w")
        self.entry_file = ttk.Entry(frame_file, width=20)
        self.entry_file.pack(side="left", padx=5)
        ttk.Button(frame_file, text="Browse", command=self.browse_file).pack(
            side="left", padx=5
        )
        frame_file.pack(pady=5)

        # Variable list
        frame_main = ttk.Frame(self)
        ttk.Label(frame_main, text="Select Variable:").grid(row=0, column=0)
        self.listbox = tk.Listbox(
            frame_main,
            selectmode=tk.SINGLE,
        )
        self.listbox.grid(row=1, column=0, padx=5, rowspan=6)

        # Trim eps
        ttk.Label(frame_main, text="Trim eps:").grid(row=1, column=1, padx=5)
        self.entry_eps = ttk.Entry(frame_main, width=10)
        self.entry_eps.insert(0, "0")
        self.entry_eps.grid(row=2, column=1, padx=5)

        # Buttons
        ttk.Button(frame_main, text="Show", command=self.on_plot).grid(
            row=3, column=1, padx=5
        )
        ttk.Button(frame_main, text="Fit", command=self.on_fit).grid(
            row=4, column=1, padx=5
        )
        ttk.Button(frame_main, text="Save", command=self.on_save).grid(
            row=5, column=1, padx=5
        )

        # Extended options button
        btn_ext = ttk.Button(
            frame_main, text="Options", command=self.open_options
        ).grid(row=6, column=1, padx=5)
        frame_main.pack(pady=5)

    def browse_file(self):
        f = filedialog.askopenfilename(
            filetypes=[("TecPlot files", "*.pdt"), ("All files", "*")],
            title="Select TecPlot File",
            initialdir=".",
        )
        if f:
            self.entry_file.delete(0, tk.END)
            # insert the filename into the entry
            self.path_fn = f
            f = f.split("/")[-1] if "/" in f else f.split("\\")[-1]
            self.entry_file.insert(0, f)
            self.load_data()

    def load_data(self):
        fn = self.path_fn.strip()
        self.tk_theta = self.entry_theta.get().strip()
        self.tk_energy = self.entry_energy.get().strip()
        zone_lines = self.entry_zone_lines.get()
        if not fn:
            messagebox.showerror("Error", "Please provide filename.")
            return
        try:
            self.data, self.vars = read_tecplot_data(
                fn, self.tk_theta, self.tk_energy, zone_lines_max=zone_lines
            )
            # remove tk_theta and tk_energy from vars
            self.vars = [
                v for v in self.vars if v not in (self.tk_theta, self.tk_energy)
            ]
            self.listbox.delete(0, tk.END)
            for v in self.vars:
                self.listbox.insert(tk.END, v)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def on_plot(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        if self.line_plot_var.get():
            line_plot(
                self.data, key, self.entry_theta.get(), self.entry_energy.get(), eps
            )
        else:
            plot_data(
                self.data, key, self.entry_theta.get(), self.entry_energy.get(), eps
            )

    def on_save(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        out = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
        )
        if out:
            # open new window with save options
            win = tk.Toplevel(self)
            win.title("Save Options")
            win.geometry("300x200")
            win.resizable(False, False)

            t_dist, t_energy = trim_data(self.data[key], self.data[self.tk_energy], eps)

            ttk.Label(win, text="Energy Resolution:").grid(
                row=1, column=0, padx=5, pady=5
            )
            e_res = ttk.Entry(win, width=20)
            e_res.grid(row=1, column=1, padx=5, pady=5)
            e_res.insert(0, str(len(t_energy)))
            e_res.config(state="disabled")
            ttk.Label(win, text="Theta Resolution:").grid(
                row=2, column=0, padx=5, pady=5
            )
            e_theta = ttk.Entry(win, width=20)
            e_theta.grid(row=2, column=1, padx=5, pady=5)
            e_theta.insert(0, str(len(self.data[self.tk_theta])))
            e_theta.config(state="disabled")

            sigma_entry = ttk.Entry(win, width=20)
            smooth_var = tk.BooleanVar()

            def _toggle_entry():
                state = "normal" if smooth_var.get() else "disabled"
                sigma_entry.config(state=state)
                e_res.config(state=state)
                e_theta.config(state=state)

            ttk.Label(win, text="Smooth Data").grid(row=0, column=0, padx=5, pady=5)
            ttk.Checkbutton(win, variable=smooth_var, command=_toggle_entry).grid(
                row=0, column=1, padx=5, pady=5
            )
            ttk.Label(win, text="Smoothing Sigma:").grid(
                row=3, column=0, padx=5, pady=5
            )
            sigma_entry.grid(row=3, column=1, padx=5, pady=5)
            sigma_entry.insert(0, "1.0")  # default sigma for smoothing
            sigma_entry.config(state="disabled")

            def _on_save():
                try:
                    e_res_val = int(e_res.get())
                    t_res_val = int(e_theta.get())
                    sigma_val = float(sigma_entry.get()) if smooth_var.get() else None
                except ValueError as ve:
                    messagebox.showerror("Error", f"Invalid input: {ve}")
                    return

                if sigma_val is not None:
                    # Smooth and downsample the data
                    smoothed_data, new_energy, new_theta = smooth_and_downsample(
                        t_dist,
                        t_energy,
                        self.data[self.tk_theta],
                        sigma=(sigma_val, sigma_val),
                        new_shape=(e_res_val, t_res_val),
                    )

                    save_dist(
                        smoothed_data,
                        new_theta,
                        new_energy,
                        out,
                        key=f"{key}",
                    )
                else:
                    save_dist(
                        t_dist,
                        self.data[self.tk_theta],
                        t_energy,
                        out,
                        key=f"{key}",
                    )
                win.destroy()
                messagebox.showinfo("Saved", f"Data saved to {out}")

            ttk.Button(win, text="Save", command=_on_save).grid(row=4, column=0, pady=5)
            ttk.Button(win, text="Close", command=win.destroy).grid(
                row=4, column=1, pady=5
            )
            win.transient(self)  # make it modal
            win.grab_set()  # block interaction with parent window

    def on_fit(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        angle_p_cutoff = self.fit_angle_p_cutoff.get()
        use_sine = self.fit_use_sin.get()
        verbose = self.fit_verbose.get()
        fit_function = self.fit_function.get()
        if fit_function == "vonMises":
            vm = True
        elif fit_function == "powerCosine":
            vm = False
        else:
            messagebox.showerror("Error", "Unknown fit function selected.")
            return
        n_gauss = self.fit_n_gauss.get()
        if n_gauss <= 0:
            n_gauss = None
        res = fit_data(
            self.data,
            key,
            self.entry_theta.get(),
            self.entry_energy.get(),
            eps,
            angle_p_cutoff=angle_p_cutoff,
            useSin=use_sine,
            verbose=verbose,
            vm=vm,
            alpha=self.fit_alpha.get(),
            beta=self.fit_beta.get(),
            n_gauss=n_gauss,
        )
        if res == -1:
            messagebox.showerror("Error", "Failed to fit data.")

    def open_options(self):
        win = tk.Toplevel(self)
        win.title("Options")
        win.geometry("300x500")
        win.resizable(False, False)

        # Keys
        frame_keys = ttk.Frame(win)
        ttk.Label(frame_keys, text="Parsing:").grid(row=0, columnspan=2, pady=5)
        ttk.Label(frame_keys, text="Theta Key:").grid(row=1, column=0)
        entry_t = ttk.Entry(frame_keys, textvariable=self.entry_theta, width=20)
        entry_t.grid(row=1, column=1)
        ttk.Label(frame_keys, text="Energy Key:").grid(row=2, column=0)
        entry_e = ttk.Entry(frame_keys, textvariable=self.entry_energy, width=20)
        entry_e.grid(row=2, column=1)
        ttk.Label(frame_keys, text="Zone Lines:").grid(row=3, column=0)
        entry_zone = ttk.Entry(frame_keys, textvariable=self.entry_zone_lines, width=20)
        entry_zone.grid(row=3, column=1)
        frame_keys.pack(pady=10)

        # Checkbox for line plot
        frame_plot = ttk.Frame(win)
        ttk.Label(frame_plot, text="Plot Options:").pack(anchor="center")
        chk_line_plot = ttk.Checkbutton(
            frame_plot,
            text="Line Plot",
            variable=self.line_plot_var,
        )
        chk_line_plot.pack(anchor="center", pady=5)
        frame_plot.pack(pady=10)

        frame_fit = ttk.Frame(win)
        ttk.Label(frame_fit, text="Fit Options:").grid(row=0, columnspan=2)
        ttk.Combobox(
            frame_fit,
            textvariable=self.fit_function,
            values=["powerCosine", "vonMises"],
            state="readonly",
        ).grid(row=1, columnspan=2, pady=5)
        ttk.Label(frame_fit, text="Cutoff:").grid(row=2, column=0)
        entry_power = ttk.Entry(
            frame_fit, textvariable=self.fit_angle_p_cutoff, width=20
        )
        entry_power.grid(row=2, column=1)
        ttk.Label(frame_fit, text="Alpha:").grid(row=3, column=0)
        entry_alpha = ttk.Entry(frame_fit, textvariable=self.fit_alpha, width=20)
        entry_alpha.grid(row=3, column=1)
        ttk.Label(frame_fit, text="Beta:").grid(row=4, column=0)
        entry_beta = ttk.Entry(frame_fit, textvariable=self.fit_beta, width=20)
        entry_beta.grid(row=4, column=1)
        ttk.Label(frame_fit, text="Gaussians:").grid(row=5, column=0)
        entry_n_gauss = ttk.Entry(frame_fit, textvariable=self.fit_n_gauss, width=20)
        entry_n_gauss.grid(row=5, column=1)
        chk_fit_sin = ttk.Checkbutton(
            frame_fit,
            text="Use Sine in Fit",
            variable=self.fit_use_sin,
        )
        chk_fit_sin.grid(row=6, columnspan=2, pady=5)
        chk_fit_verbose = ttk.Checkbutton(
            frame_fit,
            text="Show All Plots",
            variable=self.fit_verbose,
        )
        chk_fit_verbose.grid(row=7, columnspan=2, pady=5)
        frame_fit.pack(pady=10)

        # Done button to close
        ttk.Button(win, text="OK", command=win.destroy).pack(anchor="s", pady=5)


def main():
    parser = argparse.ArgumentParser(
        description="Tecplot distribution GUI or batch exporter"
    )
    parser.add_argument("-f", "--file", help="input Tecplot file", metavar="FILE")
    parser.add_argument(
        "-i", "--input", help="variable name to extract", metavar="VAR", default="ALL"
    )
    parser.add_argument(
        "-o", "--output", help="output filename", metavar="OUT", default="output.txt"
    )
    parser.add_argument("--eps", help="trim epsilon", type=float, default=1e-4)
    parser.add_argument("--fit", help="Fit distribution to data", action="store_true")

    args = parser.parse_args()

    # no input file → open GUI
    if not (args.file):
        root = tk.Tk()
        app = TecplotGUI(root)
        app.pack(fill="both", expand=True)
        root.mainloop()
        return

    # batch mode: load, trim, save, then exit
    # defaults for theta and energy keys
    theta_key = "THETA (DEG)"
    energy_key = "ENERGY (EV)"
    try:
        data_dict, variables = read_tecplot_data(
            args.file,
            thetaKey=theta_key,
            energyKey=energy_key,
        )

        if args.fit:
            if args.input == "ALL":
                print("Error: Variable input need for fit.")
                return

            if args.input not in variables:
                print(
                    f"Error: variable '{args.input}' not found in {args.file}",
                )
                return

            res = fit_data(data_dict, args.input, theta_key, energy_key, args.eps)
            if res == -1:
                print("Error fitting data.")
            return

        outFileName = args.output
        if outFileName.endswith(".txt"):
            outFileName = outFileName[:-4]

        if args.input == "ALL":
            keys = [v for v in variables if v not in (theta_key, energy_key)]
        else:
            keys = [args.input] if args.input in variables else []
        if not keys:
            print(
                f"Error: variable '{args.input}' not found in {args.file}",
                file=sys.stderr,
            )
            sys.exit(1)

        for key in keys:
            save_data(
                data_dict,
                key=key,
                thetaKey=theta_key,
                energyKey=energy_key,
                eps=args.eps,
                outname=outFileName + f"_{key}.txt",
            )
            print(f"Saved {key} → {outFileName}_{key}.txt")
    except Exception as e:
        print(f"Batch error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
