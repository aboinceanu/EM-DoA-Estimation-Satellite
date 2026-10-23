import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_elements = 4
freq = 4e9
c = 3e8
set_seed_angle = 72
angle_fixed = 90 - set_seed_angle

angle = np.linspace(-np.pi / 2, np.pi / 2, 1000)
idx_fixed = np.argmin(np.abs(angle - np.deg2rad(angle_fixed)))
wavelength = c / freq
d = wavelength / 2
k = 2 * np.pi / wavelength

source_x = np.sin(angle)
source_y = np.cos(angle)
source_mat = np.vstack((source_x, source_y))

Z_x = d * (np.arange(num_elements) - (num_elements - 1) / 2)
Z_y = np.zeros_like(Z_x)
Z = np.vstack((Z_x, Z_y))

phase_delay = k * (Z.T @ source_mat)
steering_matrix = np.exp(-1j * phase_delay)

def to_db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))


def gain_db(W, A=steering_matrix):
    """Power gain in dB of weights W against the steering vectors in A.

    W: (4,) for a single weight vector, or (4, 1000) with one column per angle.
    A single vector is applied to every steering vector; a matrix is applied
    column by column (column i of W against column i of A).
    """
    W = np.asarray(W)
    if W.ndim == 1:
        W = W[:, None]
    return to_db(np.abs(np.sum(W.conj() * A, axis=0)) ** 2)

def weights_A():
    """4-vector, unit power: w_A = steering vector at theta_A / sqrt(N)."""
    return steering_matrix[:, idx_fixed] / np.sqrt(num_elements)


def weights_B(cond_max=1e10):
    """(4, 1000) array, unit power (||w|| = 1) for every theta_B.

    Designed with unit response at theta_B and a null at theta_A (min-norm),
    then each column is scaled to ||w|| = 1, so the gain at theta_B is
    1/||w_unscaled||^2. Columns where the design is ill-conditioned
    (theta_B too close to theta_A) are left as NaN.
    """
    W_B = np.full((num_elements, len(angle)), np.nan, dtype=complex)
    f = np.array([1, 0])
    for i in range(len(angle)):
        C = steering_matrix[:, [i, idx_fixed]]
        G = C.conj().T @ C
        if np.linalg.cond(G) > cond_max:
            continue
        g = np.linalg.solve(G, f)
        w = C @ g
        W_B[:, i] = w / np.linalg.norm(w)
    return W_B


def uniformize(W):
    """Keep the phases of W, set every amplitude to 1/sqrt(N): ||w|| = 1 (NaN stays NaN)."""
    return np.exp(1j * np.angle(W)) / np.sqrt(num_elements)

def build_animation():
    """Create the figure, define update(), return the FuncAnimation."""
    L = 1000
    rng = np.random.default_rng(0)
    p_s = np.array([[1.0], [1.0]])
    p_n = 0.1

    frame_idx = np.arange(0, 1000, 5)
    n_frames = len(frame_idx)

    fig, (ax_raw, ax_red, ax_red_mov) = plt.subplots(
        1, 3, figsize=(14, 6), subplot_kw={'projection': 'polar'},
        constrained_layout=True)
    fig.suptitle("Bartlett / MVDR / AAR - 1 Fixed, 1 Moving Target")

    for ax, title in zip((ax_raw, ax_red, ax_red_mov),
                         ['Unreduced R', 'Reduced R (fixed satellite removed)',
                          'Reduced R (moving satellite removed)']):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        ax.set_title(title, va='bottom', pad=20)

    ax_raw.set_ylim(-11, 10)
    ax_red.set_ylim(-11, 10)
    ax_red_mov.set_ylim(-11, 10)

    bartlett_line, = ax_raw.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
    mvdr_line, = ax_raw.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
    aar_line, = ax_raw.plot(angle, np.zeros_like(angle), color='green', label='AAR')
    bartlett_line_red, = ax_red.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
    mvdr_line_red, = ax_red.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
    aar_line_red, = ax_red.plot(angle, np.zeros_like(angle), color='green', label='AAR')
    bartlett_line_red_mov, = ax_red_mov.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
    mvdr_line_red_mov, = ax_red_mov.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
    aar_line_red_mov, = ax_red_mov.plot(angle, np.zeros_like(angle), color='green', label='AAR')

    ax_raw.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1))
    ax_red.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1))
    ax_red_mov.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1))

    def simulate_R(idx_moving):
        A_s = steering_matrix[:, [idx_fixed, idx_moving]]
        S_s = np.sqrt(p_s / 2) * (rng.standard_normal((2, L)) + 1j * rng.standard_normal((2, L)))
        N = np.sqrt(p_n / 2) * (rng.standard_normal((num_elements, L)) + 1j * rng.standard_normal((num_elements, L)))

        X = A_s @ S_s + N

        R = (1 / L) * X @ (X.conj()).T
        R_inv = np.linalg.inv(R)
        return R, R_inv

    R_0, _ = simulate_R(frame_idx[0])
    I_0 = np.sum(steering_matrix.conj() * (R_0 @ steering_matrix), axis=0).real
    peaks_idx = sp.signal.find_peaks(I_0)[0]
    valid_peaks = angle[peaks_idx] > 0
    valid_idx = peaks_idx[valid_peaks]
    a1 = steering_matrix[:, valid_idx[0]]

    p_n_hat = np.linalg.eigvalsh(R_0)[:2].mean()
    p1_hat = ((a1.conj() @ R_0 @ a1).real / num_elements ** 2) - p_n_hat / num_elements
    print(np.rad2deg(angle[valid_idx]))

    R_0_red = R_0 - p1_hat * np.outer(a1, a1.conj())
    I_0_red = np.sum(steering_matrix.conj() * (R_0_red @ steering_matrix), axis=0).real
    peak_idx_red = np.argmax(I_0_red)
    a2_red = steering_matrix[:, peak_idx_red]
    p2_hat = (a2_red.conj() @ R_0_red @ a2_red).real / (num_elements ** 2) - p_n_hat / num_elements

    def update(j):
        R, R_inv = simulate_R(frame_idx[j])
        R_reduced = R - p1_hat * np.outer(a1, a1.conj())

        w, V = np.linalg.eigh(R_reduced)
        w = np.maximum(w, p_n_hat)
        R_reduced = (V * w) @ V.conj().T
        R_reduced_inv = np.linalg.inv(R_reduced)

        # Bartlett
        intensities_matched = np.sum(steering_matrix.conj() * (R @ steering_matrix), axis=0).real
        intensities_matched_red = np.sum(steering_matrix.conj() * (R_reduced @ steering_matrix), axis=0).real
        bartlett_line.set_ydata(to_db(intensities_matched / (num_elements)))
        bartlett_line_red.set_ydata(to_db(intensities_matched_red / (num_elements)))

        # MVDR
        denom_mvdr = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
        intensities_mvdr = 1 / (denom_mvdr + 1e-12)
        denom_mvdr_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
        intensities_mvdr_red = 1 / (denom_mvdr_red + 1e-12)
        mvdr_line.set_ydata(to_db(intensities_mvdr*num_elements))
        mvdr_line_red.set_ydata(to_db(intensities_mvdr_red*num_elements))

        # AAR
        num_aar = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
        denom_aar = np.sum(steering_matrix.conj() * ((R_inv @ R_inv) @ steering_matrix), axis=0).real
        intensities_aar = num_aar / (denom_aar + 1e-12)
        num_aar_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
        denom_aar_red = np.sum(steering_matrix.conj() * ((R_reduced_inv @ R_reduced_inv) @ steering_matrix),
                               axis=0).real
        intensities_aar_red = num_aar_red / (denom_aar_red + 1e-12)
        aar_line.set_ydata(to_db(intensities_aar))
        aar_line_red.set_ydata(to_db(intensities_aar_red))

        angle_mov_idx = np.argmax(intensities_matched_red)
        a2 = steering_matrix[:, angle_mov_idx]
        R_reduced_mov = R - p2_hat * np.outer(a2, a2.conj())
        u, W = np.linalg.eigh(R_reduced_mov)
        u = np.maximum(u, p_n_hat)
        R_reduced_mov = (W * u) @ W.conj().T
        R_reduced_mov_inv = np.linalg.inv(R_reduced_mov)

        intensities_matched_red_mov = np.sum(steering_matrix.conj() * (R_reduced_mov @ steering_matrix), axis=0).real
        bartlett_line_red_mov.set_ydata(to_db(intensities_matched_red_mov / (num_elements)))

        denom_mvdr_red_mov = np.sum(steering_matrix.conj() * (R_reduced_mov_inv @ steering_matrix), axis=0).real
        intensities_mvdr_red_mov = 1 / (denom_mvdr_red_mov + 1e-12)
        mvdr_line_red_mov.set_ydata(to_db(intensities_mvdr_red_mov*num_elements))

        num_aar_mov = np.sum(steering_matrix.conj() * (R_reduced_mov_inv @ steering_matrix), axis=0).real
        denom_aar_mov = np.sum(steering_matrix.conj() * ((R_reduced_mov_inv @ R_reduced_mov_inv) @ steering_matrix),
                               axis=0).real
        intensities_aar_mov = num_aar_mov / (denom_aar_mov + 1e-12)
        aar_line_red_mov.set_ydata(to_db(intensities_aar_mov))

        return [bartlett_line, mvdr_line, aar_line,
                bartlett_line_red, mvdr_line_red, aar_line_red,
                bartlett_line_red_mov, mvdr_line_red_mov, aar_line_red_mov]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    return ani  # keep a reference, otherwise the animation stops

def _gain_figure(title, ylim=(-40, 8)):
    plt.figure()
    plt.axvline(np.rad2deg(angle[idx_fixed]), ls=':', color='gray', label=r'$\theta_A$')
    plt.xlabel(r"$\theta_B$ (deg)")
    plt.ylabel("Gain (dB)")
    plt.title(title)
    plt.ylim(*ylim)
    plt.grid(True)


def run(mode="wA", plot=True):
    """mode: 'wA' | 'wB' | 'wB_uni' | 'animation'.

    Returns the gain in dB over the angle grid (or the FuncAnimation).
    """
    if mode == "wA":
        w_A = weights_A()
        g = gain_db(w_A)
        if plot:
            np.set_printoptions(precision=3)
            print(f"Weights of w_A: {w_A}")
            _gain_figure("Port 5: w_A")
            plt.plot(np.rad2deg(angle), g, label='Source B (sweep)')
            plt.axhline(g[idx_fixed], color='C1', label='Source A')
            plt.legend()
        return g

    if mode == "wB":
        W_B = weights_B()
        g = gain_db(W_B)
        if plot:
            _gain_figure("Port 6: w_B")
            plt.plot(np.rad2deg(angle), g, label='w_B')
            plt.ylim(-40, 8)
            plt.legend()
        return g

    if mode == "wB_uni":
        W_B = weights_B()
        W_uni = uniformize(W_B)
        g = gain_db(W_uni)
        if plot:
            _gain_figure("Port 6: w_B vs uniform amplitudes")
            plt.plot(np.rad2deg(angle), gain_db(W_B), label='w_B')
            plt.plot(np.rad2deg(angle), g, label='w_B uniform')
            plt.legend()
        return g

    if mode == "animation":
        return build_animation()

    raise ValueError(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    out = run("animation")
    plt.show()
    
    
    
    
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --------------------------------------------------
# 1. Load MATLAB single-dipole gain pattern
# --------------------------------------------------
# header=None is important: the first row contains data,
# not column names.
pattern = pd.read_csv("single_dipole_pattern.csv", header=None)

pattern.columns = ["azimuth", "elevation", "gain_dBi"]

# Create a regular azimuth/elevation grid
az_grid = np.sort(pattern["azimuth"].unique())
el_grid = np.sort(pattern["elevation"].unique())

gain_grid = (
    pattern.pivot(
        index="elevation",
        columns="azimuth",
        values="gain_dBi"
    )
    .reindex(index=el_grid, columns=az_grid)
    .to_numpy()
)

# Interpolate gain in dBi over the exported angular grid
gain_interpolator = RegularGridInterpolator(
    (el_grid, az_grid),
    gain_grid,
    bounds_error=False,
    fill_value=None
)

# --------------------------------------------------
# 2. Map your scan angles to MATLAB azimuth/elevation
# --------------------------------------------------
theta_deg = np.rad2deg(angle)

# Your existing directions:
# x = sin(theta), y = 0, z = cos(theta)
azimuth_deg = np.where(theta_deg < 0, 180.0, 0.0)
elevation_deg = 90.0 - np.abs(theta_deg)

query_points = np.column_stack((elevation_deg, azimuth_deg))
element_gain_dBi = gain_interpolator(query_points)

# --------------------------------------------------
# 3. Use your existing optimized beamformer weights
# --------------------------------------------------
w_A = weights_A()
W_B = weights_B()

# Array responses with isotropic elements
gain_A_port5_iso = gain_db(w_A)
gain_B_port6_iso = gain_db(W_B)

# Combined element-pattern + array-response curves.
# The MATLAB element gain is already in dBi.
gain_A_port5_dipole = gain_A_port5_iso + element_gain_dBi
gain_B_port6_dipole = gain_B_port6_iso + element_gain_dBi

# --------------------------------------------------
# 4. Plot Port 5
# --------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(theta_deg, gain_A_port5_iso, label="Isotropic elements")
plt.plot(theta_deg, gain_A_port5_dipole, label="MATLAB dipole pattern")
plt.xlabel("Source angle (degrees)")
plt.ylabel("Combined response (dB / dBi reference)")
plt.xlim(0,90)
plt.title("Port 5: isotropic vs MATLAB dipole")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --------------------------------------------------
# 5. Plot Port 6
# --------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(theta_deg, gain_B_port6_iso, label="Isotropic elements")
plt.plot(theta_deg, gain_B_port6_dipole, label="MATLAB dipole pattern")
plt.xlabel("Source angle (degrees)")
plt.ylabel("Combined response (dB / dBi reference)")
plt.title("Port 6: isotropic vs MATLAB dipole")
plt.xlim(0,90)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
