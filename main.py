import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


num_elements = 4
freq = 2.4e9
c = 3e8
set_seed_angle = 72
angle_fixed = 90 - set_seed_angle

angle = np.linspace(-np.pi/2, np.pi/2, 1000)
idx_fixed = np.argmin(np.abs(angle - np.deg2rad(angle_fixed)))
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
p_s = np.array([[1.0], [1.0]])
p_n = 0.1

frame_idx = np.arange(0, 1000, 5)
n_frames = len(frame_idx)


fig, (ax_raw, ax_red, ax_red_mov) = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={'projection': 'polar'},
                                     constrained_layout=True)
fig.suptitle("Bartlett / MVDR / AAR - 1 Fixed, 1 Moving Target")

for ax, title in zip((ax_raw, ax_red, ax_red_mov), ['Unreduced R', 'Reduced R (fixed satellite removed)', 'Reduced R (moving satellite removed)']):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(title, va='bottom', pad=20)


ax_raw.set_ylim(-17,2)
ax_red.set_ylim(-17,1)
ax_red_mov.set_ylim(-17,1)

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

# x_angles = np.rad2deg(angle[frame_idx])
# eig_hist = np.full((num_elements, n_frames), np.nan)
# eig_hist_red = np.full((num_elements, n_frames), np.nan)
#
# fig_eig, (ax_eig, ax_eig_red) = plt.subplots(2, 1, sharex=True, sharey=True)
# ax_eig.set_yscale('log')
# ax_eig.set_xlim(-90, 90)
# ax_eig.set_ylim(0.03, 10)
# ax_eig.axhline(p_n, ls='--', color='gray')
# ax_eig.axvline(angle_fixed, ls=':', color='gray')
# eig_lines = [ax_eig.plot([], [])[0] for m in range(4)]
# marker = ax_eig.axvline(x_angles[0], color='red')
#
# ax_eig_red.set_yscale('symlog', linthresh=0.01)
# ax_eig_red.set_ylim(-0.5, 10)
# ax_eig_red.axhline(p_n, ls='--', color='gray')
# ax_eig_red.axvline(angle_fixed, ls=':', color='gray')
# eig_lines_red = [ax_eig_red.plot([], [])[0] for m in range(4)]
# marker_red = ax_eig_red.axvline(x_angles[0], color='red')



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
a2_red = steering_matrix[:,peak_idx_red]
p2_hat = (a2_red.conj() @ R_0_red @ a2_red).real / (num_elements**2) - p_n_hat / num_elements

def to_db(x):
    return 10*np.log10(np.maximum(x,1e-12))

def update(j):
    R, R_inv = simulate_R(frame_idx[j])

    # eig_hist[:, j] = np.linalg.eigvalsh(R)[::-1]
    # for m, line in enumerate(eig_lines):
    #     line.set_data(x_angles, eig_hist[m])
    # marker.set_xdata([x_angles[j]]*2)
    R_reduced = R - p1_hat * np.outer(a1, a1.conj())

    # eig_hist_red[:, j] = np.linalg.eigvalsh(R_reduced)[::-1]
    # for m, line in enumerate(eig_lines_red):
    #     line.set_data(x_angles, eig_hist_red[m])
    # marker_red.set_xdata([x_angles[j]]*2)

    w, V = np.linalg.eigh(R_reduced)
    w = np.maximum(w, p_n_hat)
    R_reduced = (V*w)@V.conj().T
    R_reduced_inv = np.linalg.inv(R_reduced)

    # Bartlett
    intensities_matched = np.sum(steering_matrix.conj() * (R @ steering_matrix), axis=0).real
    intensities_matched_red = np.sum(steering_matrix.conj() * (R_reduced @ steering_matrix), axis=0).real
    bartlett_line.set_ydata(to_db(intensities_matched/(num_elements**2)))
    bartlett_line_red.set_ydata(to_db(intensities_matched_red/(num_elements**2)))

    # MVDR
    denom_mvdr = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
    intensities_mvdr = 1 / (denom_mvdr + 1e-12)
    denom_mvdr_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
    intensities_mvdr_red = 1 / (denom_mvdr_red + 1e-12)
    mvdr_line.set_ydata(to_db(intensities_mvdr))
    mvdr_line_red.set_ydata(to_db(intensities_mvdr_red))

    # AAR
    num_aar = np.sum(steering_matrix.conj() * (R_inv @ steering_matrix), axis=0).real
    denom_aar = np.sum(steering_matrix.conj() * ((R_inv @ R_inv) @ steering_matrix), axis=0).real
    intensities_aar = num_aar / (denom_aar + 1e-12)
    num_aar_red = np.sum(steering_matrix.conj() * (R_reduced_inv @ steering_matrix), axis=0).real
    denom_aar_red = np.sum(steering_matrix.conj() * ((R_reduced_inv @ R_reduced_inv) @ steering_matrix), axis=0).real
    intensities_aar_red = num_aar_red / (denom_aar_red + 1e-12)
    aar_line.set_ydata(to_db(intensities_aar/num_elements))
    aar_line_red.set_ydata(to_db(intensities_aar_red/num_elements))


    angle_mov_idx = np.argmax(intensities_matched_red)
    a2 = steering_matrix[:, angle_mov_idx]
    #p2_hat = ((a2.conj() @ R @ a2).real / num_elements ** 2) - p_n_hat / num_elements
    R_reduced_mov = R - p2_hat * np.outer(a2, a2.conj())
    u, W = np.linalg.eigh(R_reduced_mov)
    u = np.maximum(u, p_n_hat)
    R_reduced_mov = (W * u) @ W.conj().T
    R_reduced_mov_inv = np.linalg.inv(R_reduced_mov)

    intensities_matched_red_mov = np.sum(steering_matrix.conj() * (R_reduced_mov @ steering_matrix), axis=0).real
    bartlett_line_red_mov.set_ydata(to_db(intensities_matched_red_mov/(num_elements**2)))

    denom_mvdr_red_mov = np.sum(steering_matrix.conj() * (R_reduced_mov_inv @ steering_matrix), axis=0).real
    intensities_mvdr_red_mov = 1 / (denom_mvdr_red_mov + 1e-12)
    mvdr_line_red_mov.set_ydata(to_db(intensities_mvdr_red_mov))

    num_aar_mov = np.sum(steering_matrix.conj() * (R_reduced_mov_inv @ steering_matrix), axis=0).real
    denom_aar_mov = np.sum(steering_matrix.conj() * ((R_reduced_mov_inv @ R_reduced_mov_inv) @ steering_matrix), axis=0).real
    intensities_aar_mov = num_aar_mov / (denom_aar_mov + 1e-12)
    aar_line_red_mov.set_ydata(to_db(intensities_aar_mov / num_elements))

    return [bartlett_line, mvdr_line, aar_line,
            bartlett_line_red, mvdr_line_red, aar_line_red, bartlett_line_red_mov, mvdr_line_red_mov, aar_line_red_mov]


ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
plt.show()