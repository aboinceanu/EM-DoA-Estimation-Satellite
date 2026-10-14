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

fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetamin(-np.pi/2)
ax.set_thetamax(np.pi/2)
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_ylim(-.1, 1.2)
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Power (dB)')
ax.set_title('Bartlett Beamformer - 1 Fixed, 1 Moving Target', va='bottom', pad=20)

bartlett_line, = ax.plot(angle, np.zeros_like(angle), color='blue', label='Bartlett')
mvdr_line, = ax.plot(angle, np.zeros_like(angle), color='orange', label='MVDR')
ax.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1))

def update(idx_moving):

    A_s = steering_matrix[:,[idx_fixed,idx_moving]]
    S_s = np.sqrt(p_s/2) * (rng.standard_normal((2, L)) + 1j*rng.standard_normal((2, L)))
    N = np.sqrt(p_n/4) * (rng.standard_normal((4, L)) + 1j*rng.standard_normal((4, L)))

    X = A_s@S_s +N

    R = (1/L)*X@(X.conj()).T

    intensities = np.sum(steering_matrix.conj() * (R @ steering_matrix), axis=0).real
    intensities /= np.max(intensities)
    bartlett_line.set_ydata(intensities)

    denom = np.sum(steering_matrix.conj() * (np.linalg.inv(R) @ steering_matrix), axis=0).real
    intensities_mvdr = 1 / (denom + 1e-12)
    intensities_mvdr /= np.max(intensities_mvdr)
    mvdr_line.set_ydata(intensities_mvdr)
    return [bartlett_line, mvdr_line]

ani = FuncAnimation(fig, update, frames = np.arange(0,1000, 5), interval=50, blit=True)
plt.show()