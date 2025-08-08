# -*- coding: utf-8 -*-
# tsv_integrated_analyzer.py (Version 2.4 - Enhanced)

import os
import traceback
import numpy as np
import tkinter as tk
from tkinter import messagebox, LabelFrame, Radiobutton, StringVar, Checkbutton, BooleanVar, DISABLED, NORMAL, filedialog, scrolledtext
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import correlate, hilbert
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.linalg import coshm, sinhm, sqrtm

eps_0 = 8.854187817e-12
mu_0 = 1.25663706212e-6
q = 1.602176634e-19
k_B = 1.380649e-23
DEFAULT_ROUGHNESS_RMS_M = 0.05e-6
DEFAULT_TAN_DELTA_OX = 0.005
DEFAULT_N_A_CM3 = 1e15

class MaterialDatabase:
    def __init__(self, temperature=300):
        self.T = temperature
        self.materials = {
            'Cu': { 'sigma_ref': 5.8e7, 'alpha_res_temp': 0.0039, 'alpha_thermal': 16.5e-6, 'young_modulus': 110e9 },
            'SiO2': { 'eps_r': 3.9, 'tan_delta_ref': 0.002, 'beta_tan_delta_temp': 0.001, 'alpha_thermal': 0.5e-6, 'young_modulus': 70e9 },
            'Si': { 'eps_r_static': 11.7, 'alpha_thermal': 2.6e-6, 'young_modulus': 169e9, 'mu_h_ref': 0.045, 'mu_e_ref': 0.14, 'mu_h_temp_exp': -2.4, 'mu_e_temp_exp': -2.5, 'tau_ds': 0.1e-12, 'beta_ds': 0.5, 'n_i_300K': 1.0e10, 'Eg_eV': 1.12 }
        }
    def get_property(self, material, prop):
        mat = self.materials[material]
        if prop == 'sigma': return mat['sigma_ref'] * (1 - mat['alpha_res_temp'] * (self.T - 293))
        elif prop == 'tan_delta': return mat['tan_delta_ref'] * (1 + mat['beta_tan_delta_temp'] * (self.T - 293))
        elif prop == 'mu_h': return mat['mu_h_ref'] * ((self.T / 300)**mat['mu_h_temp_exp'])
        elif prop == 'mu_e': return mat['mu_e_ref'] * ((self.T / 300)**mat['mu_e_temp_exp'])
        elif prop == 'n_i':
            n_i_300K_m3 = mat['n_i_300K'] * 1e6; Eg_J = mat['Eg_eV'] * q
            return n_i_300K_m3 * (self.T/300)**1.5 * np.exp((Eg_J / (2 * k_B)) * (1/300 - 1/self.T))
        else: return mat.get(prop)
    def get_eps_si(self, omega, N_a_m3):
        mat = self.materials['Si']; tau_ds = mat['tau_ds']; beta_ds = mat['beta_ds']
        n_i_si_at_T = self.get_property('Si', 'n_i')
        # Handle intrinsic, p-type, n-type silicon
        if N_a_m3 > n_i_si_at_T * 10: # p-type
            p_hole = N_a_m3; n_electron = n_i_si_at_T**2 / (N_a_m3 + 1e-20)
        elif N_a_m3 < n_i_si_at_T / 10: # n-type (assuming very low N_a means it's effectively n or intrinsic, but N_a is acceptor conc, so this is low p, potentially intrinsic/n-dominated by traps/donors not modeled)
            # For simplicity, if N_a is very low, assume intrinsic-like behavior if not explicitly n-type specified.
            # In general, if N_a is extremely low, it defaults to intrinsic.
            p_hole = n_i_si_at_T; n_electron = n_i_si_at_T
        else: # near intrinsic or moderate doping
            p_hole = N_a_m3/2 + np.sqrt((N_a_m3/2)**2 + n_i_si_at_T**2); n_electron = n_i_si_at_T**2 / (p_hole + 1e-20)
        
        mu_h_si = self.get_property('Si', 'mu_h'); mu_e_si = self.get_property('Si', 'mu_e')
        sigma_si_dc = q * (p_hole * mu_h_si + n_electron * mu_e_si)
        
        # Debay-Sachs model for frequency-dependent conductivity
        # Avoid division by zero for omega=0, use a very small number
        omega_eff_for_ds = np.where(omega == 0, 1e-18, omega)
        
        # Apply Debye-Sachs model for AC conductivity
        if np.isscalar(omega): 
            sigma_ac = sigma_si_dc * ((1 - beta_ds) + beta_ds / (1 + 1j * omega_eff_for_ds * tau_ds))
        else: 
            sigma_ac = sigma_si_dc * ((1 - beta_ds) + beta_ds / (1 + 1j * omega_eff_for_ds * tau_ds))
        
        omega_eff_for_eps = np.where(omega == 0, 1e-18, omega)
        return mat['eps_r_static'] - 1j * sigma_ac / (omega_eff_for_eps * eps_0)


class TSVModel:
    def __init__(self, mode, num_tsv, height, radius, t_ox, pitch, 
                 roughness_rms, tan_delta_ox, N_a, V, V_FB, temperature, log_widget=None):
        self.mode = mode
        self.num_tsv = num_tsv
        self.height = height
        self.radius = radius
        self.t_ox = t_ox
        self.pitch = pitch if mode == "coupled" else None
        self.V = V
        self.V_FB = V_FB
        self.mat_db = MaterialDatabase(temperature)
        self.roughness_rms = roughness_rms if roughness_rms is not None else DEFAULT_ROUGHNESS_RMS_M
        self.tan_delta_ox_user = tan_delta_ox if tan_delta_ox is not None else DEFAULT_TAN_DELTA_OX
        self.N_a = N_a if N_a is not None else DEFAULT_N_A_CM3 * 1e6 # Convert to m^-3
        
        self.sigma_cu = self.mat_db.get_property('Cu', 'sigma')
        self.eps_ox_r = self.mat_db.get_property('SiO2', 'eps_r')
        
        # Temperature dependent tan_delta_ox
        base_tan_delta = self.tan_delta_ox_user
        temp_factor = (1 + self.mat_db.get_property('SiO2', 'beta_tan_delta_temp') * (self.mat_db.T - 293))
        self.tan_delta_ox = base_tan_delta * temp_factor
        
        self.eps_si_r_static = self.mat_db.get_property('Si', 'eps_r_static')
        
        self.log_widget = log_widget
        self.compute_geometry_and_depletion()

    def log(self, message):
        if self.log_widget:
            log_message(self.log_widget, message)

    def compute_geometry_and_depletion(self):
        delta_T = self.mat_db.T - 293
        alpha_cu = self.mat_db.get_property('Cu', 'alpha_thermal')
        
        # Adjust geometry for thermal expansion (simplified, assuming uniform expansion)
        self.height_adj = self.height * (1 + alpha_cu * delta_T)
        self.radius_adj = self.radius * (1 + alpha_cu * delta_T)
        
        self.r_ox = self.radius_adj + self.t_ox # Outer radius of oxide
        
        # Depletion width calculation
        V_s = self.V - self.V_FB # Surface potential
        
        if V_s <= 0: # Accumulation mode or flat-band, no depletion layer
            self.w_dep = 1e-12 # Effectively zero, but avoid division by zero
            self.log(f"  - Depletion Width: {self.w_dep * 1e9:.3f} nm (Accumulation Mode)")
        else: # Depletion or inversion mode
            # Effective Na: ensure it's not smaller than intrinsic concentration for robustness
            effective_Na = max(self.N_a, self.mat_db.get_property('Si', 'n_i') / 100) # Ensure N_a is not too small
            arg = (2 * self.eps_si_r_static * eps_0 * V_s) / (q * effective_Na + 1e-20) 
            self.w_dep = np.sqrt(arg) if arg > 0 else 1e-12
            self.log(f"  - Depletion Width: {self.w_dep * 1e9:.3f} nm (Depletion/Inversion)")
        
        # Depletion width cannot be larger than the silicon thickness (though not explicitly modeled here)
        # Add a practical cap to depletion width to avoid numerical issues or non-physical results if V_s is very high.
        # Assuming the silicon region extends significantly beyond the oxide.
        self.w_dep = min(self.w_dep, self.radius_adj * 100) # Cap at something large but finite relative to TSV size

    def compute_rlgc(self, omega):
        is_scalar = np.isscalar(omega)
        if is_scalar:
            omega = np.array([omega]) # Ensure omega is an array for vectorized operations

        eps_si_complex_rel = self.mat_db.get_eps_si(omega, self.N_a)
        eps_si_real_abs = np.real(eps_si_complex_rel) * eps_0
        eps_si_imag_abs = np.imag(eps_si_complex_rel) * eps_0 # Negative part contributes to loss

        # Resistance (R) - DC and Skin Effect
        R_dc = self.height_adj / (np.pi * self.radius_adj**2 * self.sigma_cu)
        delta_skin = np.sqrt(2 / (omega * mu_0 * self.sigma_cu + 1e-20)) # Skin depth
        
        # R_ac_base: Resistance considering skin effect. If skin depth is less than radius, use skin effect formula.
        R_ac_base = R_dc * np.where(delta_skin < self.radius_adj, self.radius_adj / (2 * delta_skin), 1)
        
        # Roughness correction (Hammerstad-Jensen model approximation)
        roughness_ratio = self.roughness_rms / (delta_skin + 1e-20)
        K_rough = 1 + (2 / np.pi) * np.arctan(1.4 * roughness_ratio**2) # Simplified Huray-like factor
        Rs = R_ac_base * K_rough

        # Inductance (L) - External and Internal
        log_arg_ext = np.maximum((self.r_ox + self.w_dep) / self.radius_adj, 1.001) # Avoid log(1) or less
        L_ext = (mu_0 / (2 * np.pi)) * np.log(log_arg_ext) * self.height_adj
        
        L_int_dc = mu_0 / (8 * np.pi) # Internal inductance at DC
        # Internal inductance decreases with skin effect. If skin depth is less than radius, it goes down.
        L_int_ac_factor = np.where(delta_skin < self.radius_adj, delta_skin / self.radius_adj, 1.0)
        L_int = L_int_dc * L_int_ac_factor * self.height_adj
        Ls = L_ext + L_int

        # Capacitance (C)
        # Oxide capacitance
        C_ox_per_unit_length = 2 * np.pi * self.eps_ox_r * eps_0 / np.log(self.r_ox / self.radius_adj)
        C_ox = C_ox_per_unit_length * self.height_adj

        # Depletion capacitance
        r_depletion_boundary = self.r_ox + self.w_dep
        if self.w_dep < 1e-12 or r_depletion_boundary <= self.r_ox + 1e-15:
            C_dep_per_unit_length = np.array([1e15]) if is_scalar else np.full_like(omega, 1e15) # Very large capacitance (short) if no depletion
        else:
            C_dep_per_unit_length = 2 * np.pi * eps_si_real_abs / np.log(r_depletion_boundary / self.r_ox)
        C_dep = C_dep_per_unit_length * self.height_adj

        # Total Series Capacitance: C_ox and C_dep are in series
        if self.V < self.V_FB: # Accumulation: Si acts as conductor, C is dominated by C_ox
            Cs = np.full_like(omega, C_ox)
        else: # Depletion/Inversion: C_ox and C_dep are in series
            C_ox_array = np.full_like(omega, C_ox) # Ensure C_ox is an array for element-wise ops
            Cs = (C_ox_array * C_dep) / (C_ox_array + C_dep + 1e-20) # Avoid division by zero

        # Conductance (G)
        # Oxide conductance (due to tan_delta)
        G_ox = omega * C_ox * self.tan_delta_ox # G = omega * C * tan_delta (for oxide)
        
        # Silicon conductance (due to Si material loss)
        if self.V < self.V_FB: # Accumulation: Si acts as conductor, high loss or short
            Gs = G_ox # Simplified, high conductivity silicon means very low impedance, not high G.
                      # In accumulation, the capacitance is dominated by C_ox, and loss is G_ox.
        else: # Depletion/Inversion
            if self.w_dep < 1e-12 or r_depletion_boundary <= self.r_ox + 1e-15:
                G_si_per_unit_length = np.zeros_like(omega) # If no depletion, no Si dielectric loss in this region
            else:
                # G = omega * C * tan_delta, where C = eps_real / (geometric factor), tan_delta = -eps_imag / eps_real
                # So G_si = omega * (eps_si_real_abs / geo_factor) * (-eps_si_imag_abs / eps_si_real_abs)
                # G_si = -omega * eps_si_imag_abs / geo_factor
                # For cylindrical: geo_factor is log(r_depletion/r_ox) / (2*pi)
                G_si_per_unit_length = 2 * np.pi * (-omega * eps_si_imag_abs) / np.log(r_depletion_boundary / self.r_ox)
            
            G_si = G_si_per_unit_length * self.height_adj
            Gs = G_ox + G_si # Total conductance is sum of oxide and silicon losses

        if self.mode == "single":
            ret = (Rs, Ls, Gs, Cs)
        else: # Coupled mode
            # Mutual Inductance (Lm)
            log_arg_lm = np.maximum(self.pitch / self.radius_adj, 1.001) # Avoid log(1)
            Lm = (mu_0 / (2 * np.pi)) * np.log(log_arg_lm) * self.height_adj
            
            # Mutual Capacitance (Cm) and Mutual Conductance (Gm) - between adjacent TSVs
            # Simplified model for Cm/Gm, usually derived from even/odd mode analysis
            # For two adjacent lines, C_m is often approximated using a parallel plate or cylindrical approximation.
            # Using a simplified formula for two wires: C_m_per_unit_length = pi * epsilon / arccosh(D/(2*r))
            # Here, D is pitch, r is r_ox (outer radius of oxide, approximate outer conductor)
            arg_cm = self.pitch / (2 * self.r_ox)
            if np.any(arg_cm <= 1 + 1e-9): # Pitch must be greater than 2*r_ox for arccosh to be real and meaningful.
                raise ValueError(f"Pitch ({self.pitch*1e6:.1f}μm) is too small for coupled mode capacitance calculation. Must be > 2 * outer oxide radius ({2*self.r_ox*1e6:.1f}μm).")
            
            Cm = (np.pi * eps_si_real_abs) / np.arccosh(arg_cm) * self.height_adj
            Gm = (np.pi * (-eps_si_imag_abs) * omega) / np.arccosh(arg_cm) * self.height_adj
            
            ret = (Rs, Ls, Gs, Cs, Lm, Cm, Gm)
        
        # If original omega was scalar, return scalar values
        if is_scalar:
            return tuple(r.item() if isinstance(r, np.ndarray) else r for r in ret)
        
        return ret

    def compute_abcd(self, omega):
        port_count = 4 if self.mode == "coupled" else 2
        abcd_total = np.zeros((len(omega), port_count, port_count), dtype=complex)
        
        for i, o in enumerate(omega):
            if self.mode == "single":
                Rs_i, Ls_i, Gs_i, Cs_i = self.compute_rlgc(o)
                Z_series = Rs_i + 1j * o * Ls_i
                Y_parallel = Gs_i + 1j * o * Cs_i
                
                # Check for near-zero impedance/admittance to avoid sqrt issues
                gamma = np.sqrt(Z_series * Y_parallel + 1e-20j) # Propagation constant
                Zc = np.sqrt(Z_series / (Y_parallel + 1e-20j)) # Characteristic impedance
                
                A = D = np.cosh(gamma * self.height_adj) # Multiply by height_adj here, not in compute_rlgc, because it's for the whole segment
                B = Zc * np.sinh(gamma * self.height_adj)
                C = np.sinh(gamma * self.height_adj) / Zc if abs(Zc) > 1e-12 else (np.sinh(gamma * self.height_adj) * 1e12) # Handle Zc near zero
                
                abcd_segment = np.array([[A, B], [C, D]])
            else: # Coupled mode
                Rs_i, Ls_i, Gs_i, Cs_i, Lm_i, Cm_i, Gm_i = self.compute_rlgc(o)
                
                # Series Impedance Matrix
                Z_matrix = np.array([[Rs_i + 1j * o * Ls_i, 1j * o * Lm_i],
                                     [1j * o * Lm_i, Rs_i + 1j * o * Ls_i]])
                
                # Parallel Admittance Matrix
                Y_diag = (Gs_i + Gm_i) + 1j * o * (Cs_i + Cm_i)
                Y_off_diag = -(Gm_i + 1j * o * Cm_i) # Negative for mutual admittance
                Y_matrix = np.array([[Y_diag, Y_off_diag],
                                     [Y_off_diag, Y_diag]])
                
                ZY = Z_matrix @ Y_matrix
                
                try:
                    # Matrix square root for propagation constant matrix
                    gamma_matrix = sqrtm(ZY) 
                    # Characteristic impedance matrix
                    # Zc = Z_matrix * inv(gamma_matrix)
                    Zc_matrix = Z_matrix @ np.linalg.inv(gamma_matrix + 1e-12 * np.eye(2))
                    
                    # Matrix hyperbolic functions (for length = height_adj)
                    Cosh_m = coshm(gamma_matrix * self.height_adj)
                    Sinh_m = sinhm(gamma_matrix * self.height_adj)
                    
                    A = Cosh_m
                    B = Sinh_m @ Zc_matrix
                    C = np.linalg.inv(Zc_matrix + 1e-12 * np.eye(2)) @ Sinh_m # Handle Zc_matrix near singular
                    D = Cosh_m
                    
                    abcd_segment = np.block([[A, B], [C, D]])
                except np.linalg.LinAlgError:
                    self.log(f"Warning: LinAlgError in matrix sqrt/inv at {o / (2*np.pi):.2f} Hz. Returning identity for segment to prevent crash.")
                    abcd_segment = np.eye(4, dtype=complex) # Fallback to identity for this frequency

            # Cascade multiple TSVs
            current_abcd = np.eye(port_count, dtype=complex)
            for _ in range(self.num_tsv):
                current_abcd = current_abcd @ abcd_segment
            
            abcd_total[i] = current_abcd
            
        return abcd_total

    def compute_s_params(self, freq_hz, z0=50):
        omega = 2 * np.pi * freq_hz
        abcd = self.compute_abcd(omega)
        s_params = np.zeros_like(abcd)

        for i in range(len(freq_hz)):
            if self.mode == "single":
                s_params[i] = abcd_to_s(abcd[i], z0)
            else:
                s_params[i] = abcd4_to_s4(abcd[i], z0)
        return s_params

def abcd_to_s(abcd, z0=50):
    A, B, C, D = abcd.ravel() # Flatten 2x2 matrix to get A, B, C, D
    
    # Denominator (avoid division by zero)
    denom = A * z0 + B + C * z0**2 + D * z0 + 1e-15 
    
    s11 = (A * z0 + B - C * z0**2 - D * z0) / denom
    s12 = 2 * (A * D - B * C) * z0 / denom
    s21 = 2 * z0 / denom
    s22 = (-A * z0 + B - C * z0**2 + D * z0) / denom
    
    return np.array([[s11, s12], [s21, s22]])

def abcd4_to_s4(abcd, z0=50):
    n = 2 # 2 ports for each side (total 4 ports)
    A_block, B_block, C_block, D_block = abcd[:n,:n], abcd[:n,n:], abcd[n:,:n], abcd[n:,n:]
    
    Z0_mat = np.eye(n) * z0
    Z0_inv_mat = np.eye(n) / z0
    
    try:
        # M_sum = A + B*Z0_inv + C*Z0 + D
        M_sum_inv = np.linalg.inv(A_block + B_block @ Z0_inv_mat + C_block @ Z0_mat + D_block + 1e-12 * np.eye(n))
        
        s_out = np.zeros((4,4), dtype=complex)
        
        # S11 = (A + B*Z0_inv - C*Z0 - D) * M_sum_inv
        s_out[0:n, 0:n] = (A_block + B_block @ Z0_inv_mat - C_block @ Z0_mat - D_block) @ M_sum_inv
        
        # S12 = 2*Z0 * M_sum_inv
        s_out[0:n, n:2*n] = 2 * Z0_mat @ M_sum_inv
        
        # S21 = 2*M_sum_inv * Z0 (this is incorrect in original, should be 2*M_sum_inv if Z0_matrix is applied on input side for S21 = Vin/Vout, but S-params are defined as sqrt(P_out/P_in). The 2*Z0_mat is important for power normalization)
        # For S21 and S12, the definition for multi-port is often:
        # S12 = (A - B*Z0_inv + C*Z0 - D)^-1 * (2*Z0)
        # S21 = 2*M_sum_inv * Z0 (No, it's typically S21 = 2 * inv(A + B/Z0 + C*Z0 + D) )
        # Let's stick to the common transformation matrix:
        # S = (A*Z0 + B - C*Z0^2 - D*Z0)^-1 * (2*Z0)
        # This is essentially: (D-CZ0)(AZ0+B)^-1 + (C Z0 - D) (Z0^-1 A + B Z0^-1)^-1 (2Z0)
        # The given formula for S21 in original abcd_to_s is 2*Z0/denom, which aligns with standard definition for single port.
        # For multi-port, it's more complex. The provided formula is a common matrix form (similar to Z = Z0(I+S)(I-S)^-1).
        # S_out[n:2*n, 0:n] (S21, which is S_receiver_ports,sender_ports)
        # S_out[n:2*n, 0:n] should be (Z0^-1 A + Z0^-1 B Z0^-1 + C + D Z0^-1)^-1 * (2*Z0^-1)
        # The provided formula S_out[n:2*n, 0:n] = 2 * M_sum_inv is commonly used for S21.
        
        s_out[n:2*n, 0:n] = 2 * M_sum_inv # S21 for multi-port (common definition)
        
        # S22 = (-A + B*Z0_inv - C*Z0 + D) * M_sum_inv
        s_out[n:2*n, n:2*n] = (-A_block + B_block @ Z0_inv_mat - C_block @ Z0_mat + D_block) @ M_sum_inv
        
        return s_out
    except np.linalg.LinAlgError:
        # Log error for debugging
        # self.log(f"Warning: LinAlgError in S-param calculation. Returning zeros.") # No self.log here
        return np.zeros((4,4), dtype=complex) # Return zeros to indicate failure

def create_gaussian_filter(win_len, sigma):
    x = np.linspace(-(win_len - 1) / 2., (win_len - 1) / 2., win_len)
    gauss = np.exp(-0.5 * (x / sigma)**2)
    return gauss / np.sum(gauss)

def generate_gold_sequence(length):
    # m-sequence LFSR helper
    def lfsr(taps, state, nbits):
        seq = []
        for _ in range(nbits):
            output_bit = state[0]
            seq.append(output_bit)
            feedback_bit = 0
            # XOR feedback bits
            for tap_idx in taps:
                feedback_bit ^= state[tap_idx - 1] # tap_idx is 1-based
            state = [feedback_bit] + state[:-1] # Shift
        return seq

    m_val = 7 # For a length of 2^m - 1 = 127
    
    # Primitive polynomials for m=7
    # x^7 + x^3 + 1 (taps: 7, 3)
    # x^7 + x^3 + x^2 + x + 1 (taps: 7, 3, 2, 1)
    # These are common pairs for Gold codes
    poly1_taps = [7, 3]
    poly2_taps = [7, 3, 2, 1] 

    state1 = [1] * m_val # Initial state for LFSR1 (all ones)
    state2 = [1] * m_val # Initial state for LFSR2 (all ones)

    # Generate full m-sequences
    seq_len = max(length, 2**m_val - 1) # Ensure we generate enough for requested length
    seq1_full = lfsr(poly1_taps, state1, seq_len)
    seq2_full = lfsr(poly2_taps, state2, seq_len)

    # Generate Gold sequence by XORing two m-sequences
    gold_seq = np.array([(a + b) % 2 for a, b in zip(seq1_full, seq2_full)])

    return gold_seq[:length] # Truncate to desired length

def enforce_causality(s_param_vector_original, freq_hz_original, dc_value):
    # Ensure frequencies are sorted
    sorted_indices = np.argsort(freq_hz_original)
    freq_sorted = freq_hz_original[sorted_indices]
    s_vec_sorted = s_param_vector_original[sorted_indices]

    max_freq_for_hilbert = np.max(freq_sorted)
    # Determine the number of points for uniform grid for Hilbert transform
    # Should be a power of 2 for FFT efficiency, and sufficiently dense
    num_points_hilbert_grid = 2**int(np.ceil(np.log2(len(freq_sorted) * 4))) # At least 4x original points
    if num_points_hilbert_grid < 4096: # Minimum practical points for reliable Hilbert transform
        num_points_hilbert_grid = 4096

    uniform_freq_for_hilbert = np.linspace(0, max_freq_for_hilbert, num_points_hilbert_grid)

    # Add DC point if not present at the beginning
    freq_for_interp_source = np.insert(freq_sorted, 0, 0.0) if freq_sorted[0] > 1e-9 else freq_sorted
    s_vec_for_interp_source = np.insert(s_vec_sorted, 0, dc_value) if freq_sorted[0] > 1e-9 else s_vec_sorted

    # Ensure DC value is correctly set if 0 Hz is already in freq_sorted
    if freq_sorted[0] <= 1e-9:
        s_vec_for_interp_source[freq_for_interp_source == 0] = dc_value

    # Interpolate real and imaginary parts separately
    interp_func_real = interp1d(freq_for_interp_source, np.real(s_vec_for_interp_source), kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_func_imag = interp1d(freq_for_interp_source, np.imag(s_vec_for_interp_source), kind='cubic', bounds_error=False, fill_value="extrapolate")

    s_uniform_real = interp_func_real(uniform_freq_for_hilbert)
    s_uniform_imag = interp_func_imag(uniform_freq_for_hilbert)
    s_uniform = s_uniform_real + 1j * s_uniform_imag
    
    # Apply Kramers-Kronig (Hilbert Transform) to log magnitude for minimum phase
    log_abs_s_uniform = np.log(np.abs(s_uniform) + 1e-12) # Add small epsilon to prevent log(0)

    # Hilbert transform for minimum phase. Note: hilbert returns analytic signal, imag part is Hilbert transform.
    # The phase_min = -imag(hilbert(log|S|)) applies to a one-sided spectrum for causal real functions.
    # Here, log|S| is a real even function of frequency, its Hilbert transform would be real and odd.
    # This standard approach of phase = -imag(hilbert(log|S|)) implies the use for minimum phase transfer functions.
    phase_min_approx_uniform = -np.imag(hilbert(log_abs_s_uniform))
    
    causal_s_param_on_uniform_grid = np.abs(s_uniform) * np.exp(1j * phase_min_approx_uniform)
    
    return uniform_freq_for_hilbert, causal_s_param_on_uniform_grid


def perform_signal_integrity_analysis(freq_hz, s_channel, data_rate_gbps, rise_fall_time_ps, num_bits, rj_sigma_ps=1.0, dj_pp_ps=5.0):
    is_coupled = (s_channel.shape[1] == 4)
    bit_period = 1.0 / (data_rate_gbps * 1e9) # in seconds
    samples_per_bit = 256 # Resolution of the waveform
    time_step = bit_period / samples_per_bit
    num_samples = num_bits * samples_per_bit
    time_vector = np.arange(num_samples) * time_step

    # Generate ideal input bit stream
    bits = generate_gold_sequence(num_bits)
    input_waveform_ideal = np.repeat(bits, samples_per_bit).astype(float) # Repeat each bit to samples_per_bit

    # Apply rise/fall time
    if rise_fall_time_ps > 0:
        sigma_t = (rise_fall_time_ps * 1e-12) / 2.5 # Approximate sigma from rise time (10%-90% rise time ~ 2.5*sigma)
        sigma_samples = sigma_t / time_step
        win_len = int(sigma_samples * 8) # Filter window length (e.g., 4 sigma on each side)
        if win_len % 2 == 0: win_len += 1 # Ensure odd length for symmetric filter
        if win_len < 3: win_len = 3 # Minimum filter length
        gauss_filter = create_gaussian_filter(win_len, sigma_samples)
        input_waveform_ideal = np.convolve(input_waveform_ideal, gauss_filter, mode='same')

    # Add Jitter
    rj = np.random.normal(0, rj_sigma_ps * 1e-12, num_bits) # Random Jitter (Gaussian)
    dj_amplitude = dj_pp_ps * 1e-12 / 2 # Deterministic Jitter (peak-to-peak converted to amplitude)
    dj = dj_amplitude * np.sin(2 * np.pi * np.arange(num_bits) / 10) # Sinusoidal DJ for simplicity
    
    # Cumulative jitter across bits
    cum_jitter = np.cumsum(rj + dj)
    
    # Interpolate jitter to sample rate
    time_for_jitter_points = np.arange(num_bits) * bit_period
    interp_jitter_func = interp1d(time_for_jitter_points, cum_jitter, kind='linear', 
                                  bounds_error=False, fill_value=(cum_jitter[0], cum_jitter[-1]) if num_bits > 0 else 0)
    cum_jitter_repeated = interp_jitter_func(time_vector)
    
    time_jittered = time_vector + cum_jitter_repeated
    
    # Interpolate ideal waveform to jittered time axis
    input_interp_func = interp1d(time_vector, input_waveform_ideal, kind='linear', 
                                 bounds_error=False, fill_value=(input_waveform_ideal[0], input_waveform_ideal[-1]))
    input_waveform = input_interp_func(time_jittered)

    # FFT of input waveform
    n_fft = len(time_vector)
    fft_freqs = np.fft.fftfreq(n_fft, d=time_step)
    vin_f = np.fft.fft(input_waveform)

    # Interpolate S-parameters to FFT frequencies and enforce causality
    def get_interp_s_inner(s_param_vector_for_port, param_type='S21'):
        # Determine DC value for causality enforcement.
        # For S11, S22, S33, S44 (reflection coefficients), DC is typically 1 (if open) or -1 (if short), or dependent on termination.
        # For S21, S31, S41 (transmission coefficients), DC is often 1 (0dB loss) if direct connection, or 0 if AC coupling/lossy.
        # Here, for TSV, expect S21/S31 close to 0dB at DC (or very low frequency), and S11 close to 0 (if matched) or -inf (if open circuit at input).
        # A more robust DC value for S-parameters depends on the type.
        # For S21/S31, a good initial guess is 1.0 (0dB loss) at DC.
        # For S11/S22, a good initial guess for matched port is 0.0 (-inf dB loss).
        if param_type == 'S21' or param_type == 'S31':
            dc_value = s_param_vector_for_port[np.abs(freq_hz) < 1e6].mean() if np.any(np.abs(freq_hz) < 1e6) else 1.0 + 0j # Take average of low freq points or assume 1.0
        elif param_type in ['S11', 'S22', '33', '44']: # Reflection coefficients
            dc_value = s_param_vector_for_port[np.abs(freq_hz) < 1e6].mean() if np.any(np.abs(freq_hz) < 1e6) else 0.0 + 0j # Assume 0.0 for matched at DC
        else: # Crosstalk, etc. Assume 0 at DC
            dc_value = 0.0 + 0j
        
        # Enforce causality
        uniform_freq_for_hilbert, causal_s_param_on_uniform_grid = enforce_causality(s_param_vector_for_port, freq_hz, dc_value)
        
        # Interpolate the causal S-parameters onto FFT frequency grid
        interp_func_real = interp1d(uniform_freq_for_hilbert, np.real(causal_s_param_on_uniform_grid), kind='cubic', bounds_error=False, fill_value=0.0)
        interp_func_imag = interp1d(uniform_freq_for_hilbert, np.imag(causal_s_param_on_uniform_grid), kind='cubic', bounds_error=False, fill_value=0.0)
        
        s_full = np.zeros_like(fft_freqs, dtype=complex)
        
        # Positive frequencies
        pos_fft_freqs_mask = fft_freqs >= 0
        s_full[pos_fft_freqs_mask] = interp_func_real(fft_freqs[pos_fft_freqs_mask]) + 1j * interp_func_imag(fft_freqs[pos_fft_freqs_mask])
        
        # Negative frequencies (conjugate symmetry for real time-domain signal)
        neg_fft_freqs_mask = fft_freqs < 0
        s_full[neg_fft_freqs_mask] = np.conj(interp_func_real(np.abs(fft_freqs[neg_fft_freqs_mask])) + 1j * interp_func_imag(np.abs(fft_freqs[neg_fft_freqs_mask])))
        
        # Ensure DC component is real (or has zero imaginary part)
        dc_idx = np.where(fft_freqs == 0)[0]
        if len(dc_idx) > 0:
            s_full[dc_idx] = np.real(s_full[dc_idx])
            
        return s_full

    # Get the interpolated S-parameters for the thru path and FEXT path
    if is_coupled:
        s_thru_interp = get_interp_s_inner(s_channel[:, 2, 0], param_type='S31') # S31 for coupled thru
        s41_interp = get_interp_s_inner(s_channel[:, 3, 0], param_type='S41') # S41 for FEXT
    else:
        s_thru_interp = get_interp_s_inner(s_channel[:, 1, 0], param_type='S21') # S21 for single thru
    
    # Calculate output waveform and FEXT waveform
    vout_f = vin_f * s_thru_interp
    output_waveform = np.real(np.fft.ifft(vout_f))
    
    fext_waveform = None
    if is_coupled:
        vfext_f = vin_f * s41_interp
        fext_waveform = np.real(np.fft.ifft(vfext_f))

    # Calculate propagation delay using cross-correlation
    delay = 0
    if len(output_waveform) > 0 and len(input_waveform) > 0:
        # Normalize for correlation
        input_norm = input_waveform / (np.max(np.abs(input_waveform)) + 1e-9)
        output_norm = output_waveform / (np.max(np.abs(output_waveform)) + 1e-9)
        
        corr = correlate(output_norm, input_norm, mode='full')
        peak_idx = np.argmax(corr)
        
        # Sub-sample interpolation for delay (parabolic interpolation around peak)
        if peak_idx > 0 and peak_idx < len(corr) - 1:
            y1, y2, y3 = corr[peak_idx-1], corr[peak_idx], corr[peak_idx+1]
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-12:
                peak_idx_float = peak_idx + 0.5 * (y1 - y3) / denom
            else:
                peak_idx_float = float(peak_idx)
        else:
            peak_idx_float = float(peak_idx)
            
        delay_idx_float = peak_idx_float - (len(input_waveform) - 1)
        delay = delay_idx_float * time_step

    return time_vector, input_waveform, output_waveform, fext_waveform, bit_period, delay, samples_per_bit, time_step

def analyze_eye_diagram(output_waveform, samples_per_bit, bit_period, num_bits, time_step):
    """
    Analyzes the output waveform to generate an eye diagram and compute key metrics.
    Version 2.4: Added Eye Width and collected crossing times for Bathtub curve proxy.
    """
    output_waveform = np.asarray(output_waveform)
    
    # Check for invalid input, but don't error on low amplitude
    if output_waveform is None or len(output_waveform) < 2 * samples_per_bit:
        return {"error": "Insufficient waveform data for eye analysis."}, None, None
        
    num_full_uis_available = len(output_waveform) // samples_per_bit
    settling_uis = 10  # Settle for 10 bits

    if num_full_uis_available <= settling_uis + 5: # Need settling + a few bits to form meaningful eye
        return {"error": f"Insufficient bits for analysis. Need > {settling_uis + 5}, Have: {num_full_uis_available}."}, None, None

    # Form the eye matrix from 2-UI segments, used for plotting the eye overlay
    folded_voltage_matrix = []
    for i in range(settling_uis, num_full_uis_available - 1): # Go up to the second to last UI
        start_idx = i * samples_per_bit
        end_idx = start_idx + 2 * samples_per_bit # 2 UIs per segment for plotting
        if end_idx <= len(output_waveform):
            segment = output_waveform[start_idx:end_idx]
            folded_voltage_matrix.append(segment)

    if not folded_voltage_matrix:
        return {"error": "Could not form any eye segments."}, None, None

    folded_voltage_matrix = np.array(folded_voltage_matrix)
    
    # Calculate statistics
    v_high_est = np.percentile(output_waveform, 95) # Robust high level
    v_low_est = np.percentile(output_waveform, 5)  # Robust low level
    
    # Use a robust calculation for eye height based on the central portion of the eye
    # This slices the middle of the eye (e.g., 0.9 to 1.1 UI of a 2-UI segment, or around 0.45 to 0.55 UI of 1-UI)
    # The eye is typically 'open' near the center of the UI.
    # For a 2-UI folded matrix (0 to 2*bit_period), the eye center would be at 0.5*bit_period and 1.5*bit_period.
    # We want the *vertical* opening, so it doesn't matter much *where* in the horizontal axis we slice,
    # as long as it's a stable 'open' part. Let's sample around the 1-UI mark.
    eye_center_slice_start = int(0.9 * samples_per_bit) # Start index relative to 0 of 2UI segment
    eye_center_slice_end = int(1.1 * samples_per_bit)   # End index relative to 0 of 2UI segment

    if eye_center_slice_end > folded_voltage_matrix.shape[1]: # Adjust if slice goes beyond array bounds
        eye_center_slice_end = folded_voltage_matrix.shape[1]
    
    eye_center_slice = folded_voltage_matrix[:, eye_center_slice_start:eye_center_slice_end]
    
    if eye_center_slice.size == 0:
        v1_mean = v_high_est
        v0_mean = v_low_est
    else:
        # v1_mean: Take the 10th percentile of voltages above the median (top part of the eye)
        v1_mean = np.percentile(eye_center_slice[eye_center_slice > np.median(eye_center_slice)], 10)
        # v0_mean: Take the 90th percentile of voltages below the median (bottom part of the eye)
        v0_mean = np.percentile(eye_center_slice[eye_center_slice < np.median(eye_center_slice)], 90)

    eye_height = max(0, v1_mean - v0_mean)
    optimal_decision_voltage = (v_high_est + v_low_est) / 2 # Simple average, can be optimized for BER

    # --- Eye Width and Bathtub Data Collection ---
    all_crossing_times_ps = np.array([])
    eye_width_ps = 0

    if eye_height > 0.001: # Only calculate eye width/bathtub if eye is meaningfully open
        # We need to find crossing points at the optimal_decision_voltage
        # For each 2-UI segment in folded_voltage_matrix:
        crossing_times_raw = [] # Store all crossing times in samples (relative to segment start)
        for segment in folded_voltage_matrix:
            # Shift segment values so that optimal_decision_voltage becomes the zero-crossing reference
            segment_shifted = segment - optimal_decision_voltage

            # Find indices where sign changes (potential crossings)
            # np.diff returns an array of len N-1. If element i is nonzero, it means segment[i] and segment[i+1] have different signs.
            # So, the actual crossing is between index i and i+1.
            crossing_indices = np.where(np.diff(np.sign(segment_shifted)))[0]

            for idx in crossing_indices:
                v1 = segment_shifted[idx]
                v2 = segment_shifted[idx + 1]
                
                if abs(v2 - v1) > 1e-9: # Avoid division by zero for flat lines
                    # Linear interpolation for sub-sample precision
                    # cross = idx + (0 - v1) / (v2 - v1)
                    interp_offset = -v1 / (v2 - v1)
                    crossing_time_sample = idx + interp_offset
                    crossing_times_raw.append(crossing_time_sample)
        
        # Convert crossing times from samples to ps and normalize to a single UI
        # A 2-UI segment is from 0 to 2*bit_period.
        # We are interested in crossings that form the eye opening.
        # These typically occur around 0.5 UI and 1.5 UI relative to the start of a 2-UI segment.
        # Modulo by samples_per_bit to fold all crossings into a single UI (0 to bit_period)
        if len(crossing_times_raw) > 0:
            all_crossing_times_samples_mod_ui = np.array(crossing_times_raw) % samples_per_bit
            all_crossing_times_ps = all_crossing_times_samples_mod_ui * time_step * 1e12 # Convert to ps

            # --- Simple Eye Width Calculation from Crossing Times ---
            # For a proper eye diagram, the "left" crossing times are generally clustered near the start of the UI (e.g., 0ps or slightly after).
            # The "right" crossing times are generally clustered near the end of the UI (e.g., bit_period_ps).
            # The eye opening is in between these two clusters.
            # We need to identify these two clusters. A simple way is to use a threshold (e.g., bit_period/2) or K-means clustering.
            
            # Let's use a simpler percentile-based method to estimate eye width
            if len(all_crossing_times_ps) > 50: # Ensure enough data points for statistics
                # Separate into two rough clusters based on time
                half_bit_period_ps = (bit_period / 2) * 1e12
                left_cluster_crossings = all_crossing_times_ps[all_crossing_times_ps < half_bit_period_ps]
                right_cluster_crossings = all_crossing_times_ps[all_crossing_times_ps >= half_bit_period_ps]

                if len(left_cluster_crossings) > 0 and len(right_cluster_crossings) > 0:
                    # Eye width is roughly (90th percentile of left cluster) - (10th percentile of right cluster).
                    # This needs to be swapped for eye width to be positive: rightmost boundary of left cluster
                    # and leftmost boundary of right cluster.
                    # e.g., for a 1-UI eye, left side crossing is low-to-high, right side crossing is high-to-low.
                    # The "eye" is open between these two crossing distributions.
                    # A common approach for eye width is to take the point where BER hits a certain value (e.g., 1e-6 or 1e-12).
                    # For a simplified approach, we can take the difference between:
                    # 90th percentile of crossings that happen early in the UI (left edge)
                    # and 10th percentile of crossings that happen late in the UI (right edge).
                    # This assumes a typical eye where "left edge" crossings are earlier than "right edge" crossings.
                    
                    # For a two-UI eye in the matrix, the crossings forming the *left* boundary of the eye are likely high-to-low
                    # crossings near 0.5*bit_period. The crossings forming the *right* boundary are low-to-high
                    # crossings near 1.5*bit_period.
                    
                    # Modulo operation for `all_crossing_times_ps` already folds them into one UI.
                    # So, `left_cluster_crossings` would be the left edge of the eye, `right_cluster_crossings` the right edge.
                    # To calculate the width, we take the rightmost part of the left distribution and the leftmost part of the right distribution.
                    # For example, the 95th percentile of left crossings and 5th percentile of right crossings.
                    
                    left_edge_for_width = np.percentile(left_cluster_crossings, 95) # 95th percentile of the left crossing cluster
                    right_edge_for_width = np.percentile(right_cluster_crossings, 5) # 5th percentile of the right crossing cluster
                    
                    eye_width_ps = max(0, right_edge_for_width - left_edge_for_width) # Ensure non-negative

    stats = {
        'eye_height': eye_height,
        'eye_width': eye_width_ps, # New metric
        'ber_estimate': 0.5, # Placeholder, true BER needs more advanced method
        'optimal_decision_voltage': optimal_decision_voltage,
        'warning': None,
        'all_crossing_times_ps': all_crossing_times_ps # For bathtub plot proxy
    }
    
    # Heuristic: If the calculated eye height is less than 1mV, it's physically closed.
    if eye_height < 0.001: 
        stats['warning'] = f"Physically Closed Eye (Height < 1mV). Plot shows residual noise/crosstalk."
        stats['eye_width'] = 0 # If height is zero, width is also zero.

    return stats, folded_voltage_matrix, all_crossing_times_ps # Always return the matrix and crossing data for plotting


def calculate_il_bandwidth(freq_hz, s_channel, mode, threshold_db):
    """
    Calculates the Insertion Loss (IL) bandwidth.
    Finds the highest frequency where S21 (single) or S31 (coupled) first drops below threshold_db.
    """
    if mode == "single":
        s_param_mag_db = 20 * np.log10(np.abs(s_channel[:, 1, 0]) + 1e-12) # S21
    else:
        s_param_mag_db = 20 * np.log10(np.abs(s_channel[:, 2, 0]) + 1e-12) # S31

    # Find where the magnitude falls below the threshold
    # We are looking for the *first* time it drops below when sweeping from low to high frequency.
    # If the curve oscillates, we want the highest frequency where it *remains* above threshold
    # before permanently dropping below. A simple approach is to find the first index.
    
    # Ensure frequencies are sorted (they should be from np.linspace)
    # Find indices where IL is less than the threshold
    below_threshold_indices = np.where(s_param_mag_db < threshold_db)[0]

    if len(below_threshold_indices) == 0:
        return freq_hz[-1] # Never drops below, so bandwidth is up to max frequency
    
    # Find the first frequency that drops below the threshold
    # The bandwidth frequency is the one immediately *before* this drop, or interpolated.
    first_drop_idx = below_threshold_indices[0]
    
    if first_drop_idx == 0: # If it's already below at the first frequency (0.1 GHz)
        return 0.0 # Effectively no bandwidth above DC threshold
        
    # Interpolate to find the exact frequency
    f1, f2 = freq_hz[first_drop_idx-1], freq_hz[first_drop_idx]
    s1_db, s2_db = s_param_mag_db[first_drop_idx-1], s_param_mag_db[first_drop_idx]
    
    if abs(s2_db - s1_db) < 1e-9: # Avoid division by zero if flat
        il_bw_freq = f1 # Take the frequency where it was just above or at threshold
    else:
        # Linear interpolation: y = y1 + (y2-y1)/(x2-x1) * (x-x1)
        # We want x when y = threshold_db
        # x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        il_bw_freq = f1 + (threshold_db - s1_db) * (f2 - f1) / (s2_db - s1_db)
        
    return il_bw_freq


def write_touchstone(s_params, freq_hz, filename, z0=50):
    n_ports = s_params.shape[1] # Number of ports (2 for single, 4 for coupled)
    
    with open(filename, 'w') as f:
        # Header for Touchstone file (GHz, S-parameters, Magnitude/Angle, Reference Impedance)
        f.write(f'# GHz S MA R {z0}\n')
        f.write(f'! TSV S-parameters for {n_ports}-port network\n')
        f.write('! Freq     ')
        for j in range(n_ports):
            for k in range(n_ports):
                f.write(f'S{j+1}{k+1}_Mag  S{j+1}{k+1}_Ang  ')
        f.write('\n')

        for i, f_hz in enumerate(freq_hz):
            row = [f_hz / 1e9] # Frequency in GHz
            for j in range(n_ports):
                for k in range(n_ports):
                    mag = np.abs(s_params[i, j, k])
                    angle = np.angle(s_params[i, j, k], deg=True)
                    row.extend([mag, angle])
            f.write(' '.join(map(lambda x: f"{x:.8e}", row)) + '\n') # Format with scientific notation

def log_message(log_widget, message):
    if not log_widget: return
    log_widget.config(state=NORMAL)
    log_widget.insert(tk.END, message + "\n")
    log_widget.see(tk.END) # Scroll to end
    log_widget.config(state=DISABLED)
    log_widget.update_idletasks() # Update GUI immediately

def compute_and_analyze(fig, canvas, log_widget, run_button):
    run_button.config(state=DISABLED, text="Analyzing...") # Disable button during analysis
    try:
        log_message(log_widget, "========================================")
        log_message(log_widget, "Starting New Analysis (v2.4 - ECTC Enhanced)...")
        
        # Get parameters from GUI
        params = get_parameters(log_widget)
        # Unpack parameters:
        # (mode, num_tsv, height, radius, t_ox, pitch, roughness_rms, tan_delta_ox, N_a, V, V_FB, temperature, 
        # start_freq, end_freq, num_points, data_rate_gbps, rise_fall_time_ps, num_bits, il_threshold_db)
        (mode, num_tsv, _, _, _, _, _, _, _, _, _, _, 
         start_freq, end_freq, num_points, data_rate_gbps, rise_fall_time_ps, num_bits, il_threshold_db) = params
        
        # Warnings for potential issues
        if end_freq > 100e9:
            log_message(log_widget, "WARNING: End frequency > 100 GHz. Expect significant signal loss and potentially a closed eye.")
        if end_freq / (data_rate_gbps * 1e9) > 10:
             log_message(log_widget, "WARNING: Simulation bandwidth is much higher than the data rate's Nyquist frequency. This is valid but may show a closed eye.")
        if num_points < end_freq / 1e9: # Heuristic: at least 1 point per GHz for good resolution
            log_message(log_widget, f"WARNING: Low number of frequency points ({num_points}) for a wide range ({end_freq/1e9} GHz). Consider increasing 'Num Points' for accuracy.")
        
        log_message(log_widget, f"Parameters validated: Mode={mode}, Total TSVs={num_tsv}, Rate={data_rate_gbps}Gbps")
        log_message(log_widget, "Initializing TSV physical model...")
        
        # Initialize TSVModel with the first 12 parameters
        tsv_model = TSVModel(*params[:12], log_widget=log_widget)
        
        log_message(log_widget, f"Computing frequency response ({start_freq/1e9:.1f} to {end_freq/1e9:.1f} GHz)...")
        freq_hz = np.linspace(start_freq, end_freq, num_points)
        s_channel = tsv_model.compute_s_params(freq_hz)
        
        # --- NEW: Calculate Insertion Loss Bandwidth ---
        il_bandwidth_freq = calculate_il_bandwidth(freq_hz, s_channel, mode, il_threshold_db)
        log_message(log_widget, f"Calculated IL Bandwidth @ {il_threshold_db:.0f}dB: {il_bandwidth_freq/1e9:.2f} GHz")

        log_message(log_widget, f"Performing transient analysis for {num_bits} bits...")
        transient_results = perform_signal_integrity_analysis(freq_hz, s_channel, *params[15:18]) # Data rate, rise/fall, num_bits
        
        log_message(log_widget, "Analyzing eye diagram...")
        # Unpack transient results
        _, _, output_waveform, _, bit_period, _, samples_per_bit, time_step = transient_results[:8]
        
        # Pass output_waveform, samples_per_bit, bit_period, num_bits, time_step to analyze_eye_diagram
        stats, eye_matrix, all_crossing_times_ps_for_bathtub = analyze_eye_diagram(output_waveform, samples_per_bit, bit_period, num_bits, time_step)
        
        log_message(log_widget, "Visualizing results in GUI...")
        
        # Pass il_bandwidth_freq to visualize_results in stats for display
        stats['il_bandwidth_freq'] = il_bandwidth_freq
        stats['il_threshold_db'] = il_threshold_db # Also pass the threshold for label
        
        visualize_results(fig, canvas, log_widget, tsv_model, freq_hz, s_channel, 
                          stats, eye_matrix, all_crossing_times_ps_for_bathtub, *transient_results)
        
        log_message(log_widget, "Prompting for S-parameter file save...")
        default_extension = f".s{2 if mode == 'single' else 4}p"
        file_types = [(f"Touchstone {'2' if mode == 'single' else '4'}-port", f"*.s{2 if mode == 'single' else 4}p"), ("All files", "*.*")]
        filename = filedialog.asksaveasfilename(initialdir=".", title="Save S-Parameter File", 
                                                defaultextension=default_extension, 
                                                filetypes=file_types, 
                                                initialfile=f"tsv_{mode}_{data_rate_gbps}Gbps.s{2 if mode == 'single' else 4}p")
        if filename:
            write_touchstone(s_channel, freq_hz, filename)
            log_message(log_widget, f"SUCCESS: S-params saved to {os.path.basename(filename)}")
        else:
            log_message(log_widget, "Info: S-parameter file save cancelled.")
        
        log_message(log_widget, "ANALYSIS COMPLETE.")
        
    except ValueError as e:
        error_msg = f"INPUT ERROR: {e}"
        log_message(log_widget, error_msg)
        messagebox.showerror("Input Error", str(e))
    except Exception as e:
        error_msg = f"ERROR: An unexpected error occurred: {e}\n{traceback.format_exc()}"
        log_message(log_widget, error_msg)
        messagebox.showerror("Error", error_msg)
    finally:
        run_button.config(state=NORMAL, text="Compute, Analyze & Save") # Re-enable button

def get_parameters(log_widget):
    try:
        use_roughness = use_roughness_var.get()
        use_tan_delta = use_tan_delta_var.get()
        use_N_a = use_N_a_var.get()

        mode = analysis_mode.get()
        num_tsv_segments = int(entry_num_tsv.get()) # Number of TSV segments in cascade
        height = float(entry_height.get()) * 1e-6 # um to m
        radius = float(entry_radius.get()) * 1e-6 # um to m
        t_ox = float(entry_t_ox.get()) * 1e-6 # um to m
        pitch = float(entry_pitch.get()) * 1e-6 if mode == "coupled" else 0 # um to m

        roughness_rms = float(entry_roughness.get()) * 1e-6 if use_roughness else None # um to m
        tan_delta_ox = float(entry_tan_delta_ox.get()) if use_tan_delta else None
        N_a_cm3 = float(entry_N_a.get()) if use_N_a else None
        
        V = float(entry_V.get())
        V_FB = float(entry_V_FB.get())
        temperature = float(entry_temperature.get()) + 273.15 # C to Kelvin

        start_freq = float(entry_start_freq.get()) * 1e9 # GHz to Hz
        end_freq = float(entry_end_freq.get()) * 1e9 # GHz to Hz
        num_points = int(entry_num_points.get())
        
        data_rate_gbps = float(entry_data_rate.get())
        rise_fall_time_ps = float(entry_rise_fall.get())
        num_bits = int(entry_num_bits.get())
        
        il_threshold_db = float(entry_il_threshold.get()) # New parameter

        N_a = N_a_cm3 * 1e6 if N_a_cm3 is not None else None # cm^-3 to m^-3

        # Input validation
        if mode == "coupled" and pitch <= 2 * (radius + t_ox):
            raise ValueError(f"Pitch ({pitch*1e6:.1f}μm) must be > 2 * outer radius ({2*(radius+t_ox)*1e6:.1f}μm).")
        if num_tsv_segments < 1 or num_points < 10 or num_bits < 50:
            raise ValueError("Invalid parameters: Num TSVs < 1, Num Points < 10, or Num Bits < 50.")
        if start_freq >= end_freq:
            raise ValueError("Start frequency must be < end frequency.")
        if data_rate_gbps <= 0 or rise_fall_time_ps < 0:
            raise ValueError("Data rate must be positive, rise/fall time non-negative.")
        
        return (mode, num_tsv_segments, height, radius, t_ox, pitch, 
                roughness_rms, tan_delta_ox, N_a, V, V_FB, temperature, 
                start_freq, end_freq, num_points, data_rate_gbps, 
                rise_fall_time_ps, num_bits, il_threshold_db)

    except ValueError as e:
        log_message(log_widget, f"PARAMETER ERROR: {e}")
        messagebox.showerror("Input Error", str(e))
        raise # Re-raise to stop execution flow


def visualize_results(fig, canvas, log_widget, tsv_model, freq_hz, s_channel, 
                      stats, eye_matrix, all_crossing_times_ps_for_bathtub, *transient_results):
    """
    Visualizes the simulation results.
    Version 2.4: Added IL Bandwidth marker, Eye Width, and Crossing Time Distribution (Bathtub proxy).
    """
    # Unpack transient results
    (time_vector, input_waveform, output_waveform, fext_waveform, bit_period, delay, _, time_step) = transient_results[:8]
    
    fig.clear()
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2) # Create a 2x2 grid of subplots
    
    fig.suptitle(f'TSV Analysis: {tsv_model.mode.title()} ({tsv_model.num_tsv} TSV, {float(entry_data_rate.get())} Gbps)', fontsize=14)
    
    # Plot 1: Frequency Response (S-parameters)
    freq_plot_ghz = freq_hz / 1e9
    ax1.set_title('Frequency Response')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True)
    ax1.set_ylim(-100, 5) # Reasonable range for S-parameter magnitudes
    
    if tsv_model.mode == "single":
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 1, 0]) + 1e-12), label='S21 (Insertion Loss)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 0, 0]) + 1e-12), '--', label='S11 (Return Loss)')
    else: # Coupled mode
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 2, 0]) + 1e-12), label='S31 (Thru)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 0, 0]) + 1e-12), '--', label='S11 (Return)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 1, 0]) + 1e-12), '-.', label='S21 (NEXT)') # Near-End Crosstalk
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 3, 0]) + 1e-12), ':', label='S41 (FEXT)') # Far-End Crosstalk
    
    # --- NEW: Add IL Bandwidth marker ---
    if 'il_bandwidth_freq' in stats and stats['il_bandwidth_freq'] is not None:
        il_bw_freq_ghz = stats['il_bandwidth_freq'] / 1e9
        il_threshold_db = stats['il_threshold_db']
        ax1.axvline(il_bw_freq_ghz, color='red', linestyle=':', 
                    label=f'IL BW @ {il_threshold_db:.0f}dB: {il_bw_freq_ghz:.2f} GHz')
        
    ax1.legend(loc='lower left', fontsize='small')
    
    # Plot 2: RLGC Parameters
    ax2.set_title('RLGC Parameters')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Value (log scale)')
    ax2.grid(True, which="both") # Grid for both major and minor ticks
    ax2.set_xscale('log') # Log scale for frequency
    ax2.set_yscale('log') # Log scale for RLGC values
    
    # Ensure RLGC calculation is performed on the same frequency range as S-params for consistency
    rlgc_data = tsv_model.compute_rlgc(2 * np.pi * freq_hz)
    R, L, G, C = rlgc_data[0:4] # Extract R, L, G, C for plotting
    
    ax2.plot(freq_plot_ghz, R, label='R (Ohm/m)')
    ax2.plot(freq_plot_ghz, L, label='L (H/m)')
    ax2.plot(freq_plot_ghz, G, label='G (S/m)')
    ax2.plot(freq_plot_ghz, C, label='C (F/m)')
    ax2.legend(fontsize='small')

    # Plot 3: Transient Waveforms
    ax3.set_title('Transient Waveforms')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True)
    
    ax3.plot(time_vector * 1e9, input_waveform, alpha=0.6, label='Input')
    ax3.plot(time_vector * 1e9, output_waveform, label=f'Output')
    if fext_waveform is not None:
        ax3.plot(time_vector * 1e9, fext_waveform, label='FEXT')
    
    ax3.legend(fontsize='small')
    # Limit x-axis to show a few bit periods for clarity
    ax3.set_xlim(0, min(len(time_vector) * time_step * 1e9, 20 * bit_period * 1e9)) # Show max 20 bit periods
    # Auto-adjust y-axis with some padding
    ax3.set_ylim(min(-0.2, np.min(output_waveform)-0.1) if len(output_waveform)>0 else -0.2, 
                 max(1.2, np.max(output_waveform)+0.1) if len(output_waveform)>0 else 1.2)


    # Plot 4: Eye Diagram & Bathtub Proxy
    ax4.set_xlabel('Time (ps)')
    ax4.set_ylabel('Voltage (V)')
    ax4.grid(True)
    
    # Check for errors or if the matrix is valid for plotting
    if 'error' in stats:
        ax4.set_title('Eye Diagram (Error)')
        ax4.text(0.5, 0.5, f'Eye Not Available\n({stats["error"]})', ha='center', va='center', color='red', transform=ax4.transAxes)
    elif eye_matrix is not None and len(eye_matrix) > 0:
        samples_per_bit = int(bit_period / time_step) # Re-calculate samples_per_bit
        
        # Center the eye diagram on the plot. A 2-UI eye will span from -0.5*UI to 1.5*UI or 0 to 2*UI.
        # Plot time from 0 to 2 * bit_period in ps
        eye_time_axis_ps = np.arange(eye_matrix.shape[1]) * time_step * 1e12
        
        # Plot the eye segments
        # Only plot a subset of segments for performance and clarity if many are available
        for i in range(min(len(eye_matrix), 500)): # Max 500 segments to plot
            ax4.plot(eye_time_axis_ps, eye_matrix[i], color='#9467bd', alpha=0.05) # Faint purple lines
        
        # Add optimal decision voltage line
        ax4.axhline(stats['optimal_decision_voltage'], color='red', linestyle='--', lw=1, label='Decision Voltage')

        # --- NEW: Plot the crossing time distribution (Bathtub proxy) ---
        if all_crossing_times_ps_for_bathtub is not None and len(all_crossing_times_ps_for_bathtub) > 0:
            # Create a second Y-axis for the histogram
            ax_hist = ax4.twinx() 
            
            # Use bins covering one UI (0 to bit_period_ps)
            hist_bins = np.linspace(0, bit_period * 1e12, 100) # 100 bins for smoother histogram
            
            # Plot histogram (probability density)
            n, bins, patches = ax_hist.hist(all_crossing_times_ps_for_bathtub, bins=hist_bins, 
                                            density=True, alpha=0.4, color='green', label='Crossing Dist.')
            
            ax_hist.set_ylabel('Probability Density', color='green')
            ax_hist.tick_params(axis='y', labelcolor='green')
            ax_hist.set_ylim(bottom=0) # Ensure histogram starts from 0
            
            # Adjust zorder to ensure eye plot is visible over histogram background
            ax4.set_zorder(ax_hist.get_zorder() + 1) # Put eye plot on top
            ax4.patch.set_visible(False) # Make background of ax4 transparent
            ax_hist.legend(loc='upper right', fontsize='small')

        # If there is a warning (e.g., physically closed eye), display it on the plot
        if stats.get('warning'):
            ax4.text(0.5, 0.95, stats['warning'], ha='center', va='top', color='orange', weight='bold', 
                     transform=ax4.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Set X-axis to show one UI, centered on the main eye opening
        # A 2-UI folded eye typically centers the eye at 1 * bit_period. So from 0.5 to 1.5 * bit_period
        # Let's show from 0 to 1 * bit_period for simplified view, as the crossing times are modulo 1 UI
        ax4.set_xlim(0, bit_period * 1e12) # Display one UI (0 to 1*bit_period) for eye width context
        
        # Auto-adjust y-axis for very small signals, but with a minimum sensible range
        min_v, max_v = np.min(eye_matrix), np.max(eye_matrix)
        v_range = max(abs(max_v - min_v), 0.01) # Ensure a minimum y-range of 10mV
        ax4.set_ylim(min_v - 0.1 * v_range, max_v + 0.1 * v_range)
        
        # Update Eye Diagram title with metrics
        ax4.set_title(f'Eye Diagram (Height: {stats["eye_height"] * 1000:.1f} mV, Width: {stats["eye_width"]:.1f} ps)')

    else:
        ax4.set_title('Eye Diagram (No Data)')
        ax4.text(0.5, 0.5, 'Eye Not Available\n(Unknown Error)', ha='center', va='center', color='red', transform=ax4.transAxes)

    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96]) # Adjust layout to prevent overlap
    canvas.draw()
    
    # Log results
    log_message(log_widget, "\n--- Analysis Results ---")
    if 'error' in stats:
        log_message(log_widget, f"  - Eye Diagram Error: {stats['error']}")
    else:
        if stats.get('warning'):
            log_message(log_widget, f"  - WARNING: {stats['warning']}")
        log_message(log_widget, f"  - Eye Height: {stats['eye_height'] * 1000:.4f} mV") # Display in mV for clarity
        log_message(log_widget, f"  - Eye Width: {stats['eye_width']:.4f} ps") # Display in ps for clarity (NEW)
        log_message(log_widget, f"  - Optimal Decision Voltage: {stats['optimal_decision_voltage']:.4f} V")
        log_message(log_widget, f"  - Propagation Delay: {delay*1e12:.2f} ps")
        
        if 'il_bandwidth_freq' in stats and stats['il_bandwidth_freq'] is not None:
            log_message(log_widget, f"  - IL Bandwidth @ {stats['il_threshold_db']:.0f}dB: {stats['il_bandwidth_freq']/1e9:.2f} GHz (NEW)")
            
    log_message(log_widget, "------------------------\n")

# --- GUI Creation and Main Loop (Modified for new parameter) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Integrated TSV Signal Integrity Analyzer v2.4 (ECTC Enhanced)")
    root.geometry("1400x900") # Wider window for better layout
    
    # Layout configuration
    root.columnconfigure(1, weight=3) # Plot frame takes more width
    root.rowconfigure(0, weight=1) # Main row takes all height

    # Control frame (left side)
    control_frame = LabelFrame(root, text="Controls", padx=10, pady=10)
    control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

    # Output frame (right side)
    output_frame = tk.Frame(root)
    output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    output_frame.rowconfigure(0, weight=5) # Plotting area
    output_frame.rowconfigure(1, weight=2) # Log area
    output_frame.columnconfigure(0, weight=1)

    # Plotting area
    plot_frame = LabelFrame(output_frame, text="Analysis Plots", padx=5, pady=5)
    plot_frame.grid(row=0, column=0, sticky="nsew")
    
    fig = Figure(figsize=(10, 8), dpi=100) # Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame) # Toolbar for plots
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Log area
    log_frame = LabelFrame(output_frame, text="Log & Results", padx=5, pady=5)
    log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    log_widget = scrolledtext.ScrolledText(log_frame, state=DISABLED, wrap=tk.WORD, height=10)
    log_widget.pack(expand=True, fill="both")

    def toggle_param_entry(var, entry, label):
        """Helper to enable/disable entry based on checkbox."""
        if var.get():
            entry.config(state=NORMAL)
            label.config(fg='black')
        else:
            entry.config(state=DISABLED)
            label.config(fg='gray')

    def toggle_mode():
        """Helper to enable/disable pitch entry based on analysis mode."""
        if analysis_mode.get() == "coupled":
            entry_pitch.config(state=NORMAL)
            label_pitch.config(fg='black')
        else:
            entry_pitch.config(state=DISABLED)
            label_pitch.config(fg='gray')

    # --- GUI Components ---
    # Analysis Mode Frame
    mode_frame = LabelFrame(control_frame, text="Analysis Mode", padx=5, pady=5)
    mode_frame.pack(fill="x", pady=5)
    analysis_mode = StringVar(value="single")
    Radiobutton(mode_frame, text="Single", variable=analysis_mode, value="single", command=toggle_mode).pack(side="left")
    Radiobutton(mode_frame, text="Coupled", variable=analysis_mode, value="coupled", command=toggle_mode).pack(side="left", padx=10)
    
    label_pitch = tk.Label(mode_frame, text="Pitch(μm):")
    label_pitch.pack(side="left")
    entry_pitch = tk.Entry(mode_frame, width=8)
    entry_pitch.insert(0, "15")
    entry_pitch.pack(side="left")

    # TSV Parameters Frame
    param_frame = LabelFrame(control_frame, text="TSV Physical Parameters", padx=5, pady=5)
    param_frame.pack(fill="x", pady=5)

    # Fixed parameters
    fixed_params = {
        "Height (μm)": "30",
        "Radius (μm)": "2.5",
        "T Ox (μm)": "0.1",
        "V_bias (V)": "1.0",
        "V_FB (V)": "-0.8",
        "Temperature (C)": "25"
    }
    entries = {}
    row_idx = 0
    for text, val in fixed_params.items():
        tk.Label(param_frame, text=text).grid(row=row_idx, column=0, columnspan=2, sticky='w', pady=1)
        entries[text] = tk.Entry(param_frame, width=15)
        entries[text].insert(0, val)
        entries[text].grid(row=row_idx, column=2, padx=5)
        row_idx += 1
    
    # Assign specific entry widgets to variables for easy access
    entry_height, entry_radius, entry_t_ox, entry_V, entry_V_FB, entry_temperature = [entries[k] for k in fixed_params]

    # Optional parameters with checkboxes
    optional_params = {
        "Roughness RMS (μm)": "0.015",
        "Tan Delta (Oxide)": "0.002",
        "N_a (cm^-3)": "1e16" # Acceptor concentration for Si doping
    }
    use_roughness_var = BooleanVar(value=True)
    use_tan_delta_var = BooleanVar(value=True)
    use_N_a_var = BooleanVar(value=True)
    optional_vars = [use_roughness_var, use_tan_delta_var, use_N_a_var]
    optional_labels = [] # To store labels for toggling color

    for i, ((text, val), var) in enumerate(zip(optional_params.items(), optional_vars)):
        chk = Checkbutton(param_frame, variable=var)
        chk.grid(row=row_idx, column=0, sticky='w')
        lbl = tk.Label(param_frame, text=text)
        lbl.grid(row=row_idx, column=1, sticky='w')
        optional_labels.append(lbl)
        
        entry = tk.Entry(param_frame, width=15)
        entry.insert(0, val)
        entry.grid(row=row_idx, column=2, padx=5)
        
        # Assign entry widgets
        if "Roughness" in text:
            entry_roughness = entry
        elif "Tan Delta" in text:
            entry_tan_delta_ox = entry
        else: # N_a
            entry_N_a = entry
        
        # Link checkbox command to toggle function
        chk.config(command=lambda v=var, e=entry, l=lbl: toggle_param_entry(v, e, l))
        row_idx += 1

    # Simulation Parameters Frame
    si_frame = LabelFrame(control_frame, text="Simulation Parameters", padx=5, pady=5)
    si_frame.pack(fill="x", pady=5)

    sim_params = {
        "Start Freq (GHz)": "0.1",
        "End Freq (GHz)": "300",
        "Num Points": "601",
        "Number of TSVs": "1", # Total number of identical TSV segments cascaded
        "Data Rate (Gbps)": "40",
        "Rise/Fall Time (ps)": "10",
        "Num Bits": "8192",
        "IL Threshold (dB)": "-20" # NEW: Insertion Loss Threshold for Bandwidth
    }
    sim_entries = {}
    for i, (text, val) in enumerate(sim_params.items()):
        tk.Label(si_frame, text=text).grid(row=i, column=0, sticky='w', pady=1)
        sim_entries[text] = tk.Entry(si_frame, width=10)
        sim_entries[text].insert(0, val)
        sim_entries[text].grid(row=i, column=1, padx=5)
    
    # Assign specific entry widgets for simulation parameters
    entry_start_freq, entry_end_freq, entry_num_points, entry_num_tsv, \
    entry_data_rate, entry_rise_fall, entry_num_bits, entry_il_threshold = [sim_entries[k] for k in sim_params]

    # Run Button
    run_button = tk.Button(control_frame, text="Compute, Analyze & Save", font=('Helvetica', 12, 'bold'))
    run_button.config(command=lambda: compute_and_analyze(fig, canvas, log_widget, run_button))
    run_button.pack(pady=20, fill='x')

    # Initial setup calls
    log_message(log_widget, "Welcome to the TSV Analyzer v2.4! This version includes IL Bandwidth and Eye Width calculations.")
    toggle_mode() # Initialize pitch entry state
    # Initialize optional parameter entry states
    for var, entry, label in zip(optional_vars, 
                                 [entry_roughness, entry_tan_delta_ox, entry_N_a], 
                                 optional_labels):
        toggle_param_entry(var, entry, label)

    root.mainloop()



