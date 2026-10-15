import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_elements = 4
freq = 2.4e9
c = 3e8
set_seed_angle = 72
angle_fixed = 90 - set_seed_angle

angle = np.linspace(-np.pi/2, np.pi/2, 1000)
idx_fixed = np.argmin(np.abs(angle - np.deg2rad(angle_fixed)))
idx_moving = 200
wavelength = c/freq

d = wavelength/2
k = 2*np.pi/wavelength

source_x = np.sin(angle)
source_y = np.cos(angle)
source_mat = np.vstack((source_x, source_y))

Z_x = d * (np.arange(num_elements) - (num_elements-1)/2)
Z_y = np.zeros_like(Z_x)
Z = np.vstack((Z_x, Z_y))

phase_delay = k*(Z.T@source_mat)
steering_matrix = np.exp(-1j*phase_delay)

L = 1000
rng = np.random.default_rng(0)
p_s= np.array([[1.0], [1.0]])
p_n = 0.1

fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.2])

ax_polar = [[fig.add_subplot(gs[r, c], projection='polar') for c in range(2)] for r in range(3)]

ax_eig     = fig.add_subplot(gs[0, 2])
ax_eig_red = fig.add_subplot(gs[1, 2], sharex=ax_eig, sharey=ax_eig)

ax_bartlett, ax_bartlett_red, ax_mvdr, ax_mvdr_red, ax_aar, ax_aar_red = np.ravel(ax_polar)

for ax, title in zip(np.ravel(ax_polar), ['Bartlett', 'Bartlett Reduced', 'MVDR', 'MVDR Reduced', 'AAR', 'AAR Reduced']):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(f"{title} Beamformer - 1 Fixed, 1 Moving Target", va='bottom', pad=20)
ax_bartlett.set_ylim(0,32)
ax_mvdr.set_ylim(0,2)
ax_aar.set_ylim(0,5)
ax_bartlett_red.set_ylim(0,20)
ax_mvdr_red.set_ylim(0,1)
ax_aar_red.set_ylim(0,5)


bartlett_line, = ax_bartlett.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
mvdr_line, = ax_mvdr.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
aar_line, = ax_aar.plot(angle, np.zeros_like(angle), color='green', label='AAR')
bartlett_line_red, = ax_bartlett_red.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
mvdr_line_red, = ax_mvdr_red.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
aar_line_red, = ax_aar_red.plot(angle, np.zeros_like(angle), color='green', label='AAR')

frame_idx = np.arange(0, 1000, 5)
n_frames = len(frame_idx)
eig_hist = np.full((4, n_frames), np.nan)
x_angles = np.rad2deg(angle[frame_idx])

ax_eig.set_yscale('log')
ax_eig.set_xlim(-90, 90)
ax_eig.set_ylim(0, 10)
ax_eig.axhline(p_n, ls='--', color='gray')
ax_eig.axvline(angle_fixed, ls=':', color='gray')
eig_lines = [ax_eig.plot([], [])[0] for m in range(4)]
marker = ax_eig.axvline(x_angles[0], color='red')

eig_hist_red = np.full((4, n_frames), np.nan)
ax_eig_red.set_yscale('symlog', linthresh=0.01)
ax_eig_red.set_ylim(0, 10)
ax_eig_red.set_xlim(-90, 90)
ax_eig_red.axhline(p_n, ls='--', color='gray')
ax_eig_red.axvline(angle_fixed, ls=':', color='gray')
eig_lines_red = [ax_eig_red.plot([], [])[0] for m in range(4)]
marker_red = ax_eig_red.axvline(x_angles[0], color='red')

def update(j):
    idx_moving = frame_idx[j]
    A_s = steering_matrix[:,[idx_fixed,idx_moving]]
    S_s = np.sqrt(p_s/2) * (rng.standard_normal((2, L)) + 1j*rng.standard_normal((2, L)))
    N = np.sqrt(p_n/2) * (rng.standard_normal((4, L)) + 1j*rng.standard_normal((4, L)))

    X = A_s@S_s + N

    R = (1/L)*X@(X.conj()).T
    R_inv = np.linalg.inv(R)
    if j == 0:
        eig_hist[:] = np.nan
    eig_hist[:,j] = np.linalg.eigvalsh(R)[::-1]
    for m, line in enumerate(eig_lines):
        line.set_data(x_angles, eig_hist[m])
    marker.set_xdata([x_angles[j]]*2)
    p_n_hat = eig_hist[-2:, j].mean()
    a1 = steering_matrix[:, idx_fixed]
    p1_hat = (a1.conj() @ R @ a1).real / num_elements ** 2 - p_n_hat / num_elements
    R_reduced = R - p1_hat * np.outer(a1, a1.conj())

    if j == 0:
        eig_hist_red[:] = np.nan
    eig_hist_red[:,j] = np.linalg.eigvalsh(R_reduced)[::-1]
    for m, line in enumerate(eig_lines_red):
        line.set_data(x_angles, eig_hist_red[m])
    marker_red.set_xdata([x_angles[j]]*2)
    w, V = np.linalg.eigh(R_reduced)
    w = np.maximum(w, p_n_hat)
    R_reduced = (V*w)@V.conj().T
    R_reduced_inv = np.linalg.inv(R_reduced)

    intensities_matched = np.sum(steering_matrix.conj() * (R @ steering_matrix), axis=0).real
    intensities_matched_red = np.sum(steering_matrix.conj() * (R_reduced @ steering_matrix), axis=0).real
    bartlett_line.set_ydata(intensities_matched)
    bartlett_line_red.set_ydata(intensities_matched_red)

    denom_mvdr = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
    intensities_mvdr = 1 / (denom_mvdr + 1e-12)
    denom_mvdr_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
    intensities_mvdr_red = 1 / (denom_mvdr_red + 1e-12)
    mvdr_line.set_ydata(intensities_mvdr)
    mvdr_line_red.set_ydata(intensities_mvdr_red)

    num_aar = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
    denom_aar = np.sum(steering_matrix.conj() * ((R_inv@R_inv) @ steering_matrix), axis=0).real
    intensities_aar = num_aar / (denom_aar + 1e-12)
    num_aar_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
    denom_aar_red = np.sum(steering_matrix.conj() * ((R_reduced_inv @ R_reduced_inv) @ steering_matrix), axis=0).real
    intensities_aar_red = num_aar_red / (denom_aar_red + 1e-12)
    aar_line.set_ydata(intensities_aar)
    aar_line_red.set_ydata(intensities_aar_red)

    return [bartlett_line, mvdr_line, aar_line, bartlett_line_red, mvdr_line_red, aar_line_red, *eig_lines, marker, *eig_lines_red, marker_red]

ani = FuncAnimation(fig, update, frames = n_frames, interval=50, blit=True)
plt.show()