import numpy as np
import matplotlib.pyplot as plt


class Processor:
    def __init__(self):
        self.inputs = {name: 1.0 + 0j for name in "ABCD"}
        self.coupling_ratios = 0.5
        self.phases = {
            "phi1": 277.87,
            "phi2": 277.87,
            "phi3": 105.73,
            "phi4": 0.0,
            "phi5": 0.0,
            "psiB": 0.0,
            "psiA": -172.135,
        }

        #(input, output, gain, fixed_phase, {phase_variables, , })
        self.lines = [
            ("A", "A4", 1.0,   0.0, {"phi2": 1}),
            ("B", "B4", 1.0,   0.0, {}),

            ("A", "A3", 1.0,   0.0, {"psiA": 1}),
            ("B", "B3", 1.0,   0.0, {"psiB": 1}),

            ("A", "A2", 1.0,   0.0, {"psiA": 2, "phi1": 1}),
            ("B", "B2", 1.0,   0.0, {"psiB": 2}),

            ("A", "A1", 1.0,   0.0, {"psiA": 3}),
            ("B", "B1", 1.0,   0.0, {"psiB": 3}),
        ]

        #(source, gain, fixed_phase_deg, {phase_variables, , })
        self.stages = {
            # Stage 1: ABCD -> XYZQ
            "XYZQ": {
                "XA": [
                    ("A4", 1/np.sqrt(2), -90, {}),#{"phi2": 1}),
                    ("A3", 1/np.sqrt(2), -180, {}),
                ],
                "XB": [    
                    ("B4", 1/np.sqrt(2), -90, {}),
                    ("B3", 1/np.sqrt(2), -180, {}),
                ],
                "YA": [
                    ("A4", 1/np.sqrt(2), -180, {"phi3": 1}),#("A4", 1/np.sqrt(2), -180, {"phi2": 1, "phi3": 1}),
                    ("A3", 1/np.sqrt(2), -90, {"phi3": 1}),
                ],
                "YB": [
                    ("B4", 1/np.sqrt(2), -180, {}),
                    ("B3", 1/np.sqrt(2), -90, {}),
                ],
                "ZA": [
                    ("A2", 1/np.sqrt(2), -90, {}),#{"phi1": 1}),
                    ("A1", 1/np.sqrt(2), -180, {"phi4": 1}),
                ],
                "ZB": [
                    ("B2", 1/np.sqrt(2), -90, {"phi4": 1}),
                    ("B1", 1/np.sqrt(2), -180, {"phi4": 1}),
                ],
                "QA": [
                    ("A2", 1/np.sqrt(2), -180, {}),#{"phi1": 1}),
                    ("A1", 1/np.sqrt(2), -90, {}),
                ],
                "QB": [
                    ("B2", 1/np.sqrt(2), -180, {}),
                    ("B1", 1/np.sqrt(2), -90, {}),
                ],
            },

            # Stage 2: XYZQ -> Alpha, Beta, Gamma
            "ABGZ": {
                "Alpha_A": [
                    ("XA", 1/np.sqrt(2), -90, {}),
                    ("ZA", 1/np.sqrt(2), -180, {}),
                ],
                "Alpha_B": [
                    ("XB", 1/np.sqrt(2), -90, {"phi5": 1}),
                    ("ZB", 1/np.sqrt(2), -180, {"phi5": 1}),
                ],
                "Beta_A": [
                    ("YA", 1/np.sqrt(2), -90, {}),
                    ("QA", 1/np.sqrt(2), -180, {}),
                ],
                "Beta_B": [
                    ("YB", 1/np.sqrt(2), -90, {}),
                    ("QB", 1/np.sqrt(2), -180, {}),
                ],
                "Gamma_A": [
                    ("YA", 1/np.sqrt(2), -180, {}),
                    ("QA", 1/np.sqrt(2), -90, {}),
                ],
                "Gamma_B": [
                    ("YB", 1/np.sqrt(2), -180, {}),
                    ("QB", 1/np.sqrt(2), -90, {}),
                ],
                "Zero_A": [
                    ("XA", 1/np.sqrt(2), -180, {}),
                    ("ZA", 1/np.sqrt(2), -90, {}),
                ],
                "Zero_B": [
                    ("XB", 1/np.sqrt(2), -180, {}),
                    ("ZB", 1/np.sqrt(2), -90, {}),
                ],
            },

            # Stage 3: Alpha, Beta, Gamma -> final outputs
            "OUT": {
                "Output1_A": [
                    ("Alpha_A", np.sqrt(self.coupling_ratios), -90, {}),
                    ("Beta_A", np.sqrt(1 - self.coupling_ratios), -180, {}),
                ],
                "Output1_B": [
                    ("Alpha_B", np.sqrt(self.coupling_ratios), -90, {}),
                    ("Beta_B", np.sqrt(1 - self.coupling_ratios), -180, {}),
                ],
                "Output2_A": [
                    ("Alpha_A", np.sqrt(self.coupling_ratios), -180, {}),
                    ("Beta_A", np.sqrt(1 - self.coupling_ratios), -90, {}),
                ],
                "Output2_B": [
                    ("Alpha_B", np.sqrt(self.coupling_ratios), -180, {}),
                    ("Beta_B", np.sqrt(1 - self.coupling_ratios), -90, {}),
                ],
            },
        }
        
        
    def calculate(self):
        signals = dict(self.inputs)

        # Calculate ports 1, 2, 3, 4 from self.lines
        ports = {}

        for source, output, gain, offset, variables in self.lines:
            phase = offset + sum(
                coefficient * self.phases[name]
                for name, coefficient in variables.items()
            )

            contribution = (
                signals[source]
                * gain
                * np.exp(1j * np.deg2rad(phase))
            )

            ports[output] = ports.get(output, 0j) + contribution

        signals.update(ports)

        # Calculate each subsequent stage
        for stage in self.stages.values():
            new_signals = {}

            for output, terms in stage.items():
                value = 0j

                for source, gain, offset, variables in terms:
                    phase = offset + sum(
                        coefficient * self.phases[name]
                        for name, coefficient in variables.items()
                    )

                    value += (
                        signals[source]
                        * gain
                        * np.exp(1j * np.deg2rad(phase))
                    )

                new_signals[output] = value

            signals.update(new_signals)

        return signals

    def variable_coupling_ratio(self, ratio):
        self.coupling_ratios = float(np.clip(ratio, 0.0, 1.0))
        
        self.stages["OUT"]["Output1_A"] = [
            ("Alpha_A", np.sqrt(self.coupling_ratios), -90, {}),
            ("Beta_A", np.sqrt(1 - self.coupling_ratios), -180, {}),
        ]
        self.stages["OUT"]["Output1_B"] = [
            ("Alpha_B", np.sqrt(self.coupling_ratios), -90, {}),
            ("Beta_B", np.sqrt(1 - self.coupling_ratios), -180, {}),
        ]
        
        # I mean this output should be zero but ye you never know
        self.stages["OUT"]["Output2_A"] = [
        ("Alpha_A", np.sqrt(self.coupling_ratios), -180, {}),
        ("Beta_A", np.sqrt(1 - self.coupling_ratios), -90, {}),
        ]
        self.stages["OUT"]["Output2_B"] = [
        ("Alpha_B", np.sqrt(self.coupling_ratios), -180, {}),
        ("Beta_B", np.sqrt(1 - self.coupling_ratios), -90, {}),
        ]
    def quantize_phase_2bit(phase):
    # """Quantize a phase to the nearest 2-bit phase state."""
        phase_states = np.array([0.0, 90.0, 180.0, 270.0])
        phase = phase % 360.0

        differences = np.abs(
            (phase - phase_states + 180.0) % 360.0 - 180.0
        )

        return phase_states[np.argmin(differences)]
    
def calculate_source(p, source):
    original_inputs = p.inputs.copy()

    p.inputs = {name: 0j for name in "ABCD"}
    p.inputs[source] = 1.0 + 0j

    signals = p.calculate()

    p.inputs = original_inputs
    return signals


p = Processor()

theta_B_sweep = np.linspace(0, 90, 181)  # 0.1-degree steps
potential_ratios = np.linspace(0, 1, 201)
max_power_results = []
ratio_results = []
gain_A_port5_results = []
gain_B_port5_results = []
gain_A_port6_results = []
gain_B_port6_results = []
for angle in theta_B_sweep:
    psi_B = -np.pi * np.sin(np.deg2rad(angle))
    result_deg = np.rad2deg(psi_B)
    p.phases["psiB"] = result_deg
    powers = []
    # Find the best coupler ratio for this theta_B
    for ratio in potential_ratios:
        p.variable_coupling_ratio(ratio)
        signals_B = calculate_source(p, "B")
        power = abs(signals_B["Output1_B"])**2
        powers.append(power)
        # Store the optimal ratio and corresponding power
    best_index = np.argmax(powers)

    ratio_results.append(potential_ratios[best_index])
    max_power_results.append(powers[best_index])
    p.variable_coupling_ratio(potential_ratios[best_index])
    signals_B = calculate_source(p, "B")

    # Port 5: Gamma
    gain_B_port5_results.append(10 * np.log10(max(abs(signals_B["Gamma_B"])**2, 1e-15)))
    # Port 6: Output1
    gain_B_port6_results.append(10 * np.log10(max(abs(signals_B["Output1_B"])**2, 1e-15)))



# for name, value in signals.items():
#     print(
#         f"{name:8s} "
#         f"amplitude={abs(value):.4f}, "
#         f"phase={np.angle(value, deg=True):.2f}DEG, "
#         f"power={abs(value)**2:.4f}"
#     )

# signals_A = calculate_source(p, "A")
# signals_B = calculate_source(p, "B")

# print("A flow:")
# for name, value in signals_A.items():
#     print(f"{name:8s} amplitude={abs(value):.4f}, "
#         f"phase={np.angle(value, deg=True):.2f}°, "
#         f"power={abs(value)**2:.4f}")

# print("\nB flow:")
# for name, value in signals_B.items():
#     print(f"{name:8s} amplitude={abs(value):.4f}, "
#         f"phase={np.angle(value, deg=True):.2f}°, "
#         f"power={abs(value)**2:.4f}")
    
    

# # Plot optimal coupling ratio versus theta_B
# plt.figure(figsize=(9, 5))
# plt.plot(theta_B_sweep, ratio_results, color="blue")
# plt.xlabel(r"$\theta_B$ (degrees)")
# plt.ylabel("Optimal coupling power ratio")
# plt.title(r"Optimal Coupler Ratio vs. $\theta_B$")
# plt.xlim(0, 90)
# plt.ylim(0, 1)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Optional: plot the maximum achievable Output1 power
# plt.figure(figsize=(9, 5))
# plt.plot(theta_B_sweep, max_power_results, color="red")
# plt.xlabel(r"$\theta_B$ (degrees)")
# plt.ylabel("Maximum Output1 power")
# plt.title(r"Maximum Output1 Power vs. $\theta_B$")
# plt.grid(True)
# plt.tight_layout()
# plt.show()






# Fix theta_A = 73 degrees
def find_phi12():
    p.phases["psiA"] = 172.2

    # Sweep phi1 and phi2
    phase_values = np.linspace(0, 360, 721)  # 0.5-degree steps

    best_power = np.inf
    best_phi1 = None
    best_phi2 = None

    for phi1 in phase_values:
        p.phases["phi1"] = phi1

        for phi2 in phase_values:
            p.phases["phi2"] = phi2

            signals_A = calculate_source(p, "A")

            # Residual power after the X and Z paths combine
            residual_power = abs(signals_A["Alpha_A"])**2

            if residual_power < best_power:
                best_power = residual_power
                best_phi1 = phi1
                best_phi2 = phi2

    print(f"Optimal phi1 = {best_phi1:.2f} degrees")
    print(f"Optimal phi2 = {best_phi2:.2f} degrees")
    print(f"Residual Alpha_A power = {best_power:.6e}")
    
    p.phases["phi1"] = best_phi1
    p.phases["phi2"] = best_phi2
    signals_A = calculate_source(p, "A")

    X = signals_A["XA"]
    Z = signals_A["ZA"]

    print(f"XA: amplitude={abs(X):.6f}, "
        f"phase={np.angle(X, deg=True):.2f} degrees")
    print(f"ZA: amplitude={abs(Z):.6f}, "
        f"phase={np.angle(Z, deg=True):.2f} degrees")
    print(f"Phase difference = "
        f"{((np.angle(Z/X, deg=True) + 180) % 360) - 180:.2f} degrees")
    print(f"Alpha_A residual = {abs(signals_A['Alpha_A']):.6e}")
    
def find_phi3():
    p.phases["psiA"] = 172.2

    phi3_values = np.linspace(0, 360, 3601)
    gamma_powers = []

    for phi3 in phi3_values:
        p.phases["phi3"] = phi3
        signals_A = calculate_source(p, "A")
        gamma_powers.append(abs(signals_A["Gamma_A"])**2)

    best_index = np.argmax(gamma_powers)
    best_phi3 = phi3_values[best_index]

    p.phases["phi3"] = best_phi3

    print(f"Optimal phi3 = {best_phi3:.1f} degrees")
    print(f"Maximum Gamma_A power = "
        f"{gamma_powers[best_index]:.6f}")

    signals_A = calculate_source(p, "A")
    YA = signals_A["YA"]
    QA = signals_A["QA"]

    print(f"YA phase = {np.angle(YA, deg=True):.2f} degrees")
    print(f"QA phase = {np.angle(QA, deg=True):.2f} degrees")
    print(f"QA - YA phase difference = "
        f"{(np.angle(QA / YA, deg=True) % 360):.2f} degrees")

    return best_phi3

# best_phi3 = find_phi3()

# find_phi12()
# find_phi3()

# for phi3 in [74.4, 90.0]:
#     p.phases["phi3"] = phi3
#     signals_A = calculate_source(p, "A")

#     YA = signals_A["YA"]
#     QA = signals_A["QA"]
#     Gamma = signals_A["Gamma_A"]

#     print(f"\nphi3 = {phi3:.1f} degrees")
#     print(f"YA amplitude = {abs(YA):.6f}")
#     print(f"YA phase = {np.angle(YA, deg=True):.2f} degrees")
#     print(f"QA amplitude = {abs(QA):.6f}")
#     print(f"QA phase = {np.angle(QA, deg=True):.2f} degrees")
#     print(f"QA - YA = {(np.angle(QA / YA, deg=True) % 360):.2f} degrees")
#     print(f"Gamma_A power = {abs(Gamma)**2:.6f}")




p = Processor()
signals_A_test = calculate_source(p,"A")
print("=== AAAAAA path ==================")
for i in ["A1", "A2", "A3", "A4", "XA", "YA", "ZA", "QA", "Alpha_A", "Beta_A", "Gamma_A", "Zero_A", "Output1_A", "Output2_A"]:
    value = signals_A_test[i]
    amplitude = abs(value)
    phase = np.angle(value, deg=True)
    input_amplitude = abs(p.inputs["A"])
    output_amplitude = abs(signals_A_test[i])
    gain_db = 20 * np.log10(output_amplitude / input_amplitude)
    print(
        f"{i}: complex = {value:.4f}; "
        f"amplitude = {amplitude:.4f}; "
        f"phase = {phase:.2f};",
        (f"Amplitude gain {i} = {gain_db:.2f} dB\n")
    )



# signals_B_test = calculate_source(p,"B")
# print("\n\n=== BBBBBB path ==================")
# for j in ["B1", "B2", "B3", "B4"]:
#     value = signals_B_test[j]
#     amplitude = abs(value)
#     phase = np.angle(value, deg=True)
#     print(
#         f"{j}: complex = {value:.4f}, "
#         f"amplitude = {amplitude:.4f}, "
#         f"phase = {phase:.2f}\n"
#     )

# p.phases["phi3"] = 105.73
# signals = calculate_source(p, "A")

# print("Gamma_A =", signals["Gamma_A"])
# print("|Gamma_A| =", abs(signals["Gamma_A"]))



theta_B_sweepQ2 = np.linspace(0, 90, 181)
# port5_B_power = []
gain_dB_port5_B = []

for theta_B in theta_B_sweepQ2:
    psi_B = -np.pi * np.sin(np.deg2rad(theta_B))
    psi_B_deg = np.rad2deg(psi_B)

    # Calculate port 5 signal amplitude for source B
    port5_B = 0.5 * (
        np.exp(1j * np.deg2rad(360))
        + np.exp(-1j * np.deg2rad(psi_B_deg - 270))
        + np.exp(-1j * np.deg2rad(2 * psi_B_deg - 270))
        + np.exp(-1j * np.deg2rad(3 * psi_B_deg - 180))
    )
    input_amplitude = abs(p.inputs["B"])
    gain_dB_port5_B.append(20 * np.log10((abs(port5_B)) / input_amplitude))

    # power = abs(port5_B)**2

    # port5_B_power.append(power)
    # port5_B_gain_db.append(
    #     10 * np.log10(max(power, 1e-15))
    # )

# Plot port 5 power gain in dB
plt.figure(figsize=(9, 5))
plt.plot(theta_B_sweepQ2, gain_dB_port5_B, color="blue")
plt.xlabel(r"$\theta_B$ (degrees)")
plt.ylabel("Port 5_B gain (dB)")
plt.title(r"Port 5 Gain for Source B vs. $\theta_B$")
plt.xlim(0, 90)
plt.grid(True)
plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# theta_B_sweepQ3 = np.linspace(0, 90, 181)
# phase_values = np.arange(0, 360, 0.5)  # 2-degree resolution

# phi4_opt = []
# phi5_opt = []

# B = 1.0
# ratio = 1#p.coupling_ratios
# max_power_results = []
# for theta_B in theta_B_sweepQ3:

#     psi_B = -180 * np.sin(np.deg2rad(theta_B))

#     best_power = -np.inf
#     best_phi4 = 0
#     best_phi5 = 0

#     for phi4 in phase_values:
#         for phi5 in phase_values:

#             # Your original Output1_B expression
#             Output1_B = (B * ratio / 2) * (
#                 np.exp(-1j * np.deg2rad(-270 + phi5))
#                 + np.exp(-1j * np.deg2rad(
#                     psi_B - 360 + phi5
#                 ))
#                 + np.exp(-1j * np.deg2rad(
#                     2 * psi_B - 360 + phi4 + phi5
#                 ))
#                 + np.exp(-1j * np.deg2rad(
#                     3 * psi_B - 450 + phi4 + phi5
#                 ))
#                 + np.exp(-1j * np.deg2rad(-540))
#                 + np.exp(-1j * np.deg2rad(
#                     psi_B - 450
#                 ))
#                 + np.exp(-1j * np.deg2rad(
#                     2 * psi_B - 540
#                 ))
#                 + np.exp(-1j * np.deg2rad(
#                     3 * psi_B - 450
#                 ))
#             )

#             power = abs(Output1_B)**2

#             if power > best_power:
#                 best_power = power
#                 best_phi4 = phi4
#                 best_phi5 = phi5

#     phi4_opt.append(best_phi4)
#     phi5_opt.append(best_phi5)
#     max_power_results.append(best_power)

# # One plot, two lines
# plt.figure(figsize=(10, 6))

# plt.plot(theta_B_sweepQ3, phi4_opt, label=r"$\phi_4$", linewidth=2)
# plt.plot(theta_B_sweepQ3, phi5_opt, label=r"$\phi_5$", linewidth=2)

# plt.xlabel(r"$\theta_B$ (degrees)")
# plt.ylabel("Optimal phase (degrees)")
# plt.title(r"Optimal $\phi_4$ and $\phi_5$ vs. $\theta_B$")
# plt.xlim(0, 90)
# plt.ylim(0, 360)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(9, 4))
# plt.plot(theta_B_sweep, max_power_results)
# plt.xlabel(r"$\theta_B$ (degrees)")
# plt.ylabel(r"Maximum $|Output1_B|^2$")
# plt.title("Maximum Output1_B Power vs. Theta B")
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# theta_B_sweepQ3 = np.linspace(0, 90, 181)
# potential_ratios = np.linspace(0, 1, 201)
# phi4_opt = []
# phi5_opt = []
# ratio_opt = []
# max_power_results = []
# B = 1.0

# for theta_B in theta_B_sweepQ3:

#     psi = -180 * np.sin(np.deg2rad(theta_B))

#     # Terms in your original Output1_B expression:
#     # Output1_B = (B*ratio/2) * (exp(-j*phi5)*A + C)

#     a = (
#         np.exp(-1j * np.deg2rad(-270))
#         + np.exp(-1j * np.deg2rad(psi - 360))
#     )

#     b = (
#         np.exp(-1j * np.deg2rad(2 * psi - 360))
#         + np.exp(-1j * np.deg2rad(3 * psi - 450))
#     )

#     C = (
#         np.exp(-1j * np.deg2rad(-540))
#         + np.exp(-1j * np.deg2rad(psi - 450))
#         + np.exp(-1j * np.deg2rad(2 * psi - 540))
#         + np.exp(-1j * np.deg2rad(3 * psi - 450))
#     )

#     # Maximize |a + exp(-j*phi4)*b|
#     # Align the phases of the two complex contributions.
#     phi4 = (np.angle(b, deg=True) - np.angle(a, deg=True)) % 360

#     A = a + np.exp(-1j * np.deg2rad(phi4)) * b

#     # Maximize |exp(-j*phi5)*A + C|
#     phi5 = (np.angle(A, deg=True) - np.angle(C, deg=True)) % 360
#     # Maximum power for each possible coupling ratio
#     powers = []
#     for ratio in potential_ratios:
#         output = (B * ratio / 2) * (
#             np.exp(-1j * np.deg2rad(phi5)) * A + C
#         )
#         powers.append(abs(output)**2)

#     # Select the best coupling ratio
#     best_index = np.argmax(powers)
#     best_ratio = potential_ratios[best_index]
#     best_power = powers[best_index]
#     phi4_opt.append(phi4)
#     phi5_opt.append(phi5)
#     ratio_opt.append(best_ratio)
#     max_power_results.append(best_power)

# fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# # Subplot 1: Optimal phases
# axes[0].plot(theta_B_sweepQ3, phi4_opt, label=r"$\phi_4$")
# axes[0].plot(theta_B_sweepQ3, phi5_opt, label=r"$\phi_5$")
# axes[0].set_ylabel("Phase (degrees)")
# axes[0].set_title("Optimal Phase Settings vs. Theta B")
# axes[0].set_ylim(0, 360)
# axes[0].grid(True)
# axes[0].legend()

# # Subplot 2: Optimal coupler ratio
# axes[1].plot(theta_B_sweepQ3, ratio_opt, color="green", label="Coupler Ratio")
# axes[1].set_ylabel("Coupler Ratio")
# axes[1].set_ylim(0, 1)
# axes[1].grid(True)
# axes[1].legend()

# # Subplot 3: Maximum output power
# axes[2].plot(theta_B_sweepQ3, max_power_results, color="red", label="Max. Power")
# axes[2].set_xlabel(r"$\theta_B$ (degrees)")
# axes[2].set_ylabel(r"$|Output1_B|^2$")
# axes[2].set_title("Maximum Output Power vs. Theta B")
# axes[2].grid(True)
# axes[2].legend()

# # Common x-axis settings
# axes[2].set_xlim(0, 90)

# plt.tight_layout()
# plt.show()



#USE THIS FOR Q3###############
# plt.figure(figsize=(10, 6))
# plt.plot(theta_B_sweepQ3, phi4_opt, label=r"$\phi_4$", linewidth=2)
# plt.plot(theta_B_sweepQ3, phi5_opt, label=r"$\phi_5$", linewidth=2)

# plt.xlabel(r"$\theta_B$ (degrees)")
# plt.ylabel("Optimal phase (degrees)")
# plt.title(r"Optimal $\phi_4$ and $\phi_5$ vs. $\theta_B$")
# plt.xlim(0, 90)
# plt.ylim(0, 360)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



# # Angles chosen for verification, including the unusual 30-degree point
# test_angles = [0, 12, 23, 30, 39, 48.5, 60, 69, 75, 82, 90]

# phase_values = np.arange(0, 360, 0.5)
# B = 1.0
# ratio = p.coupling_ratios

# # Vectorized brute-force search grids
# phi4_grid = np.deg2rad(phase_values)[:, None]
# phi5_grid = np.deg2rad(phase_values)[None, :]


def compare_methods(theta_B):

    psi = -180 * np.sin(np.deg2rad(theta_B))

    # Group your original Output1_B expression
    a = (
        np.exp(-1j * np.deg2rad(-270))
        + np.exp(-1j * np.deg2rad(psi - 360))
    )

    b = (
        np.exp(-1j * np.deg2rad(2 * psi - 360))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    C = (
        np.exp(-1j * np.deg2rad(-540))
        + np.exp(-1j * np.deg2rad(psi - 450))
        + np.exp(-1j * np.deg2rad(2 * psi - 540))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    # Analytical solution
    phi4_analytical = (
        np.angle(b, deg=True) - np.angle(a, deg=True)
    ) % 360

    A = a + np.exp(-1j * np.deg2rad(phi4_analytical)) * b

    phi5_analytical = (
        np.angle(A, deg=True) - np.angle(C, deg=True)
    ) % 360

    analytical_output = (B * ratio / 2) * (
        np.exp(-1j * np.deg2rad(phi5_analytical)) * A + C
    )
    analytical_power = abs(analytical_output)**2

    # Brute-force search, vectorized for speed
    A_grid = a + np.exp(-1j * phi4_grid) * b

    output_grid = (B * ratio / 2) * (
        np.exp(-1j * phi5_grid) * A_grid + C
    )

    power_grid = np.abs(output_grid)**2

    best_index = np.unravel_index(
        np.argmax(power_grid), power_grid.shape
    )

    phi4_brute = phase_values[best_index[0]]
    phi5_brute = phase_values[best_index[1]]
    brute_power = power_grid[best_index]

    print(f"\ntheta_B = {theta_B:.1f} degrees")
    print(
        f"Analytical: phi4 = {phi4_analytical:.2f}°, "
        f"phi5 = {phi5_analytical:.2f}°, "
        f"power = {analytical_power:.8f}"
    )
    print(
        f"Brute force: phi4 = {phi4_brute:.2f}°, "
        f"phi5 = {phi5_brute:.2f}°, "
        f"power = {brute_power:.8f}"
    )
    print(
        f"Power difference = "
        f"{abs(analytical_power - brute_power):.3e}"
    )


# for angle in test_angles:
#     compare_methods(angle)




import numpy as np
import matplotlib.pyplot as plt

theta_B_sweepQ3 = np.linspace(0, 90, 181)
potential_ratios = np.linspace(0, 1, 201)
phase_values = np.arange(0, 360, 1.0)

phi4_opt = []
phi5_opt = []
ratio_opt = []
max_power_results = []

B = 1.0

# Phase grids for vectorised optimisation
phi4_grid = np.deg2rad(phase_values)[:, None]
phi5_grid = np.deg2rad(phase_values)[None, :]

for theta_B in theta_B_sweepQ3:

    psi = -180 * np.sin(np.deg2rad(theta_B))

    a = (
        np.exp(-1j * np.deg2rad(-270))
        + np.exp(-1j * np.deg2rad(psi - 360))
    )

    b = (
        np.exp(-1j * np.deg2rad(2 * psi - 360))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    C = (
        np.exp(-1j * np.deg2rad(-540))
        + np.exp(-1j * np.deg2rad(psi - 450))
        + np.exp(-1j * np.deg2rad(2 * psi - 540))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    # best_power = -np.inf
    best_gain_dB = -np.inf
    best_phi4 = 0.0
    best_phi5 = 0.0
    best_ratio = 0.0

    # Search all three parameters
    for ratio in potential_ratios:

        # Coupler amplitude coefficients, as in Processor
        alpha = np.sqrt(ratio)
        beta = np.sqrt(1 - ratio)

        # phi4-dependent contribution
        A_grid = a + np.exp(-1j * phi4_grid) * b

        # Match the relative output-path weights
        # phi5 controls the phase of the Alpha contribution
        output_grid = (B / 2) * (
            alpha * np.exp(-1j * phi5_grid) * A_grid
            + beta * C
        )

        power_grid = np.abs(output_grid)**2

        index = np.unravel_index(
            np.argmax(power_grid), power_grid.shape
        )
        # power = power_grid[index]
        power = power_grid[index]
        gain_dB = 10 * np.log10(max(power / abs(B)**2, 1e-15))

        # if power > best_power:
        #     best_power = power
        #     best_phi4 = phase_values[index[0]]
        #     best_phi5 = phase_values[index[1]]
        #     best_ratio = ratio
        if gain_dB > best_gain_dB:
            best_gain_dB = gain_dB
            best_phi4 = phase_values[index[0]]
            best_phi5 = phase_values[index[1]]
            best_ratio = ratio

    phi4_opt.append(best_phi4)
    phi5_opt.append(best_phi5)
    ratio_opt.append(best_ratio)
    # max_power_results.append(best_power)
    max_power_results.append(best_gain_dB)


fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Subplot 1: Optimal phases
axes[0].plot(theta_B_sweepQ3, phi4_opt, label=r"$\phi_4$")
axes[0].plot(theta_B_sweepQ3, phi5_opt, label=r"$\phi_5$")
axes[0].set_ylabel("Phase (degrees)")
axes[0].set_title("Optimal Phase Settings vs. Theta B")
axes[0].set_ylim(0, 360)
axes[0].grid(True)
axes[0].legend()

# Subplot 2: Optimal coupler ratio
axes[1].plot(theta_B_sweepQ3, ratio_opt, color="green", label="Coupler Ratio")
axes[1].set_ylabel("Coupler Ratio")
axes[1].set_ylim(0, 1)
axes[1].grid(True)
axes[1].legend()

# Subplot 3: Maximum output power
axes[2].plot(theta_B_sweepQ3, max_power_results, color="red", label="Max. Power")
axes[2].set_xlabel(r"$\theta_B$ (degrees)")
axes[2].set_ylabel(r"$|Output1_B|^2$")
axes[2].set_title("Maximum Output Power vs. Theta B")
axes[2].grid(True)
axes[2].legend()

# Common x-axis settings
axes[2].set_xlim(0, 90)

plt.tight_layout()
plt.show()




theta_B_sweepQ5 = np.linspace(0, 90, 181)
potential_ratios = np.linspace(0, 1, 201)

# 2-bit phase shifter states
phase_values = np.array([0.0, 90.0, 180.0, 270.0])

phi4_opt = []
phi5_opt = []
ratio_opt = []
gain_dB_results = []

B = 1.0

for theta_B in theta_B_sweepQ5:

    psi = -180 * np.sin(np.deg2rad(theta_B))

    a = (
        np.exp(-1j * np.deg2rad(-270))
        + np.exp(-1j * np.deg2rad(psi - 360))
    )

    b = (
        np.exp(-1j * np.deg2rad(2 * psi - 360))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    C = (
        np.exp(-1j * np.deg2rad(-540))
        + np.exp(-1j * np.deg2rad(psi - 450))
        + np.exp(-1j * np.deg2rad(2 * psi - 540))
        + np.exp(-1j * np.deg2rad(3 * psi - 450))
    )

    best_gain_dB = -np.inf
    best_phi4 = 0.0
    best_phi5 = 0.0
    best_ratio = 0.0

    # Search only the four available phase states
    for phi4 in phase_values:

        A = a + np.exp(-1j * np.deg2rad(phi4)) * b

        for phi5 in phase_values:

            for ratio in potential_ratios:

                alpha = np.sqrt(ratio)
                beta = np.sqrt(1 - ratio)

                output = (B / 2) * (
                    alpha * np.exp(-1j * np.deg2rad(phi5)) * A
                    + beta * C
                )

                power = abs(output)**2
                gain_dB = 10 * np.log10(
                    max(power / abs(B)**2, 1e-15)
                )

                if gain_dB > best_gain_dB:
                    best_gain_dB = gain_dB
                    best_phi4 = phi4
                    best_phi5 = phi5
                    best_ratio = ratio

    phi4_opt.append(best_phi4)
    phi5_opt.append(best_phi5)
    ratio_opt.append(best_ratio)
    gain_dB_results.append(best_gain_dB)


# Plot results
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axes[0].plot(theta_B_sweepQ5, phi4_opt, label=r"$\phi_4$")
axes[0].plot(theta_B_sweepQ5, phi5_opt, label=r"$\phi_5$")
axes[0].set_ylabel("Phase (degrees)")
axes[0].set_title("Optimal 2-bit Phase Settings vs. Theta B")
axes[0].set_yticks([0, 90, 180, 270])
axes[0].grid(True)
axes[0].legend()

axes[1].plot(theta_B_sweepQ5, ratio_opt, color="green")
axes[1].set_ylabel("Optimal Coupler Ratio")
axes[1].set_ylim(0, 1)
axes[1].grid(True)

axes[2].plot(theta_B_sweepQ5, gain_dB_results, color="red")
axes[2].set_xlabel(r"$\theta_B$ (degrees)")
axes[2].set_ylabel("Output1_B Gain (dB)")
axes[2].set_title("Maximum Gain with 2-bit Phase Shifters")
axes[2].grid(True)

axes[2].set_xlim(0, 90)
plt.tight_layout()
plt.show()
