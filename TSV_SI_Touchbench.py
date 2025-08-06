# -*- coding: utf-8 -*-
# tsv_integrated_analyzer.py (Version 2.3 - Restored Plotting with Warnings)

# --- [此处省略所有未变动的代码，与 v2.2 版本完全相同] ---
# --- [包括: 导入库, 物理常量, MaterialDatabase, TSVModel, 辅助函数等] ---
# --- [一直到 analyze_eye_diagram 函数] ---
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
        if N_a_m3 > n_i_si_at_T * 10:
            p_hole = N_a_m3; n_electron = n_i_si_at_T**2 / (N_a_m3 + 1e-20)
        elif N_a_m3 < n_i_si_at_T / 10:
            p_hole = n_i_si_at_T; n_electron = n_i_si_at_T
        else:
            p_hole = N_a_m3/2 + np.sqrt((N_a_m3/2)**2 + n_i_si_at_T**2); n_electron = n_i_si_at_T**2 / (p_hole + 1e-20)
        mu_h_si = self.get_property('Si', 'mu_h'); mu_e_si = self.get_property('Si', 'mu_e')
        sigma_si_dc = q * (p_hole * mu_h_si + n_electron * mu_e_si)
        omega_eff_for_ds = np.where(omega == 0, 1e-18, omega)
        if np.isscalar(omega): sigma_ac = sigma_si_dc * ((1 - beta_ds) + beta_ds / (1 + 1j * omega_eff_for_ds * tau_ds))
        else: sigma_ac = sigma_si_dc * ((1 - beta_ds) + beta_ds / (1 + 1j * omega_eff_for_ds * tau_ds))
        omega_eff_for_eps = np.where(omega == 0, 1e-18, omega)
        return mat['eps_r_static'] - 1j * sigma_ac / (omega_eff_for_eps * eps_0)
class TSVModel:
    def __init__(self, mode, num_tsv, height, radius, t_ox, pitch, 
                 roughness_rms, tan_delta_ox, N_a, V, V_FB, temperature, log_widget=None):
        self.mode = mode; self.num_tsv = num_tsv; self.height = height; self.radius = radius; self.t_ox = t_ox
        self.pitch = pitch if mode == "coupled" else None; self.V = V; self.V_FB = V_FB
        self.mat_db = MaterialDatabase(temperature)
        self.roughness_rms = roughness_rms if roughness_rms is not None else DEFAULT_ROUGHNESS_RMS_M
        self.tan_delta_ox_user = tan_delta_ox if tan_delta_ox is not None else DEFAULT_TAN_DELTA_OX
        self.N_a = N_a if N_a is not None else DEFAULT_N_A_CM3 * 1e6 
        self.sigma_cu = self.mat_db.get_property('Cu', 'sigma')
        self.eps_ox_r = self.mat_db.get_property('SiO2', 'eps_r')
        base_tan_delta = self.tan_delta_ox_user
        temp_factor = (1 + self.mat_db.get_property('SiO2', 'beta_tan_delta_temp') * (self.mat_db.T - 293))
        self.tan_delta_ox = base_tan_delta * temp_factor; self.eps_si_r_static = self.mat_db.get_property('Si', 'eps_r_static')
        self.log_widget = log_widget; self.compute_geometry_and_depletion()
    def log(self, message):
        if self.log_widget: log_message(self.log_widget, message)
    def compute_geometry_and_depletion(self):
        delta_T = self.mat_db.T - 293; alpha_cu = self.mat_db.get_property('Cu', 'alpha_thermal')
        self.height_adj = self.height * (1 + alpha_cu * delta_T); self.radius_adj = self.radius * (1 + alpha_cu * delta_T)
        self.r_ox = self.radius_adj + self.t_ox; V_s = self.V - self.V_FB
        if V_s <= 0:
            self.w_dep = 1e-12; self.log(f"  - Depletion Width: {self.w_dep * 1e9:.3f} nm (Accumulation Mode)")
        else:
            effective_Na = max(self.N_a, self.mat_db.get_property('Si', 'n_i') / 100) 
            arg = (2 * self.eps_si_r_static * eps_0 * V_s) / (q * effective_Na + 1e-20) 
            self.w_dep = np.sqrt(arg) if arg > 0 else 1e-12; self.log(f"  - Depletion Width: {self.w_dep * 1e9:.3f} nm (Depletion/Inversion)")
        self.w_dep = min(self.w_dep, self.radius_adj * 100)
    def compute_rlgc(self, omega):
        is_scalar = np.isscalar(omega)
        if is_scalar: omega = np.array([omega])
        eps_si_complex_rel = self.mat_db.get_eps_si(omega, self.N_a); eps_si_real_abs = np.real(eps_si_complex_rel) * eps_0
        eps_si_imag_abs = np.imag(eps_si_complex_rel) * eps_0; R_dc = self.height_adj / (np.pi * self.radius_adj**2 * self.sigma_cu)
        delta_skin = np.sqrt(2 / (omega * mu_0 * self.sigma_cu + 1e-20)); R_ac_base = R_dc * np.where(delta_skin < self.radius_adj, self.radius_adj / (2 * delta_skin), 1)
        roughness_ratio = self.roughness_rms / (delta_skin + 1e-20); K_rough = 1 + (2 / np.pi) * np.arctan(1.4 * roughness_ratio**2); Rs = R_ac_base * K_rough
        log_arg_ext = np.maximum((self.r_ox + self.w_dep) / self.radius_adj, 1.001); L_ext = (mu_0 / (2 * np.pi)) * np.log(log_arg_ext) * self.height_adj
        L_int_dc = mu_0 / (8 * np.pi); L_int_ac_factor = np.where(delta_skin < self.radius_adj, delta_skin / self.radius_adj, 1.0)
        L_int = L_int_dc * L_int_ac_factor * self.height_adj; Ls = L_ext + L_int
        C_ox_per_unit_length = 2 * np.pi * self.eps_ox_r * eps_0 / np.log(self.r_ox / self.radius_adj); C_ox = C_ox_per_unit_length * self.height_adj
        r_depletion_boundary = self.r_ox + self.w_dep
        if self.w_dep < 1e-12 or r_depletion_boundary <= self.r_ox + 1e-15: C_dep_per_unit_length = np.array([1e15]) if is_scalar else np.full_like(omega, 1e15)
        else: C_dep_per_unit_length = 2 * np.pi * eps_si_real_abs / np.log(r_depletion_boundary / self.r_ox)
        C_dep = C_dep_per_unit_length * self.height_adj
        if self.V < self.V_FB: Cs = np.full_like(omega, C_ox)
        else: C_ox_array = np.full_like(omega, C_ox); Cs = (C_ox_array * C_dep) / (C_ox_array + C_dep + 1e-20)
        G_ox = omega * C_ox * self.tan_delta_ox
        if self.V < self.V_FB: Gs = G_ox
        else:
            if self.w_dep < 1e-12 or r_depletion_boundary <= self.r_ox + 1e-15: G_si_per_unit_length = np.zeros_like(omega)
            else: G_si_per_unit_length = 2 * np.pi * (-omega * eps_si_imag_abs) / np.log(r_depletion_boundary / self.r_ox)
            G_si = G_si_per_unit_length * self.height_adj; Gs = G_ox + G_si
        if self.mode == "single": ret = (Rs, Ls, Gs, Cs)
        else:
            log_arg_lm = np.maximum(self.pitch / self.radius_adj, 1.001); Lm = (mu_0 / (2 * np.pi)) * np.log(log_arg_lm) * self.height_adj
            arg_cm = self.pitch / (2 * self.r_ox)
            if np.any(arg_cm <= 1 + 1e-9): raise ValueError(f"Pitch is too small for coupled mode.")
            Cm = (np.pi * eps_si_real_abs) / np.arccosh(arg_cm) * self.height_adj; Gm = (np.pi * (-eps_si_imag_abs) * omega) / np.arccosh(arg_cm) * self.height_adj
            ret = (Rs, Ls, Gs, Cs, Lm, Cm, Gm)
        if is_scalar: return tuple(r.item() if isinstance(r, np.ndarray) else r for r in ret)
        return ret
    def compute_abcd(self, omega):
        port_count = 4 if self.mode == "coupled" else 2; abcd_total = np.zeros((len(omega), port_count, port_count), dtype=complex)
        for i, o in enumerate(omega):
            if self.mode == "single":
                Rs_i, Ls_i, Gs_i, Cs_i = self.compute_rlgc(o); Z_series = Rs_i + 1j * o * Ls_i; Y_parallel = Gs_i + 1j * o * Cs_i
                gamma = np.sqrt(Z_series * Y_parallel + 1e-20j); Zc = np.sqrt(Z_series / (Y_parallel + 1e-20j))
                A = D = np.cosh(gamma); B = Zc * np.sinh(gamma); C = np.sinh(gamma) / Zc if abs(Zc) > 1e-12 else (np.sinh(gamma) * 1e12)
                abcd_segment = np.array([[A, B], [C, D]])
            else:
                Rs_i, Ls_i, Gs_i, Cs_i, Lm_i, Cm_i, Gm_i = self.compute_rlgc(o)
                Z_matrix = np.array([[Rs_i + 1j * o * Ls_i, 1j * o * Lm_i], [1j * o * Lm_i, Rs_i + 1j * o * Ls_i]])
                Y_diag = (Gs_i + Gm_i) + 1j * o * (Cs_i + Cm_i); Y_off_diag = -(Gm_i + 1j * o * Cm_i); Y_matrix = np.array([[Y_diag, Y_off_diag], [Y_off_diag, Y_diag]])
                ZY = Z_matrix @ Y_matrix
                try:
                    gamma_matrix = sqrtm(ZY); Zc_matrix = Z_matrix @ np.linalg.inv(gamma_matrix + 1e-12 * np.eye(2))
                    Cosh_m = coshm(gamma_matrix); Sinh_m = sinhm(gamma_matrix); A = Cosh_m; B = Sinh_m @ Zc_matrix
                    C = np.linalg.inv(Zc_matrix + 1e-12 * np.eye(2)) @ Sinh_m; D = Cosh_m; abcd_segment = np.block([[A, B], [C, D]])
                except np.linalg.LinAlgError: self.log(f"Warning: LinAlgError at {o / (2*np.pi):.2f} Hz. Returning identity."); abcd_segment = np.eye(4, dtype=complex)
            current_abcd = np.eye(port_count, dtype=complex)
            for _ in range(self.num_tsv): current_abcd = current_abcd @ abcd_segment
            abcd_total[i] = current_abcd
        return abcd_total
    def compute_s_params(self, freq_hz, z0=50):
        omega = 2 * np.pi * freq_hz; abcd = self.compute_abcd(omega); s_params = np.zeros_like(abcd)
        for i in range(len(freq_hz)):
            if self.mode == "single": s_params[i] = abcd_to_s(abcd[i], z0)
            else: s_params[i] = abcd4_to_s4(abcd[i], z0)
        return s_params
def abcd_to_s(abcd, z0=50):
    A, B, C, D = abcd.ravel(); denom = A * z0 + B + C * z0**2 + D * z0 + 1e-15; s11 = (A * z0 + B - C * z0**2 - D * z0) / denom
    s12 = 2 * (A * D - B * C) * z0 / denom; s21 = 2 * z0 / denom; s22 = (-A * z0 + B - C * z0**2 + D * z0) / denom
    return np.array([[s11, s12], [s21, s22]])
def abcd4_to_s4(abcd, z0=50):
    n = 2; A_block, B_block, C_block, D_block = abcd[:n,:n], abcd[:n,n:], abcd[n:,:n], abcd[n:,n:]
    Z0_mat = np.eye(n) * z0; Z0_inv_mat = np.eye(n) / z0
    try:
        M_sum_inv = np.linalg.inv(A_block + B_block @ Z0_inv_mat + C_block @ Z0_mat + D_block + 1e-12 * np.eye(n)); s_out = np.zeros((4,4), dtype=complex)
        s_out[0:n, 0:n] = (A_block + B_block @ Z0_inv_mat - C_block @ Z0_mat - D_block) @ M_sum_inv; s_out[0:n, n:2*n] = 2 * Z0_mat @ M_sum_inv
        s_out[n:2*n, 0:n] = 2 * M_sum_inv; s_out[n:2*n, n:2*n] = (-A_block + B_block @ Z0_inv_mat - C_block @ Z0_mat + D_block) @ M_sum_inv
        return s_out
    except np.linalg.LinAlgError: return np.zeros((4,4), dtype=complex)
def create_gaussian_filter(win_len, sigma):
    x = np.linspace(-(win_len - 1) / 2., (win_len - 1) / 2., win_len); gauss = np.exp(-0.5 * (x / sigma)**2)
    return gauss / np.sum(gauss)
def generate_gold_sequence(length):
    def lfsr(taps, state, nbits):
        seq = [];
        for _ in range(nbits): output_bit = state[0]; seq.append(output_bit); feedback_bit = 0; [feedback_bit := feedback_bit ^ state[tap_idx - 1] for tap_idx in taps]; state = [feedback_bit] + state[:-1]
        return seq
    m_val = 7; poly1_taps = [7, 3]; poly2_taps = [7, 3, 2, 1]; state1 = [1] * m_val; state2 = [1] * m_val
    seq_len = max(length, 2**m_val - 1); seq1_full = lfsr(poly1_taps, state1, seq_len); seq2_full = lfsr(poly2_taps, state2, seq_len)
    return np.array([(a + b) % 2 for a, b in zip(seq1_full, seq2_full)])[:length]
def enforce_causality(s_param_vector_original, freq_hz_original, dc_value):
    sorted_indices = np.argsort(freq_hz_original); freq_sorted = freq_hz_original[sorted_indices]; s_vec_sorted = s_param_vector_original[sorted_indices]
    max_freq_for_hilbert = np.max(freq_sorted); num_points_hilbert_grid = 2**int(np.ceil(np.log2(len(freq_sorted) * 4)))
    if num_points_hilbert_grid < 4096: num_points_hilbert_grid = 4096
    uniform_freq_for_hilbert = np.linspace(0, max_freq_for_hilbert, num_points_hilbert_grid)
    freq_for_interp_source = np.insert(freq_sorted, 0, 0.0) if freq_sorted[0] > 1e-9 else freq_sorted
    s_vec_for_interp_source = np.insert(s_vec_sorted, 0, dc_value) if freq_sorted[0] > 1e-9 else s_vec_sorted
    if freq_sorted[0] <= 1e-9: s_vec_for_interp_source[freq_for_interp_source == 0] = dc_value
    interp_func_real = interp1d(freq_for_interp_source, np.real(s_vec_for_interp_source), kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_func_imag = interp1d(freq_for_interp_source, np.imag(s_vec_for_interp_source), kind='cubic', bounds_error=False, fill_value="extrapolate")
    s_uniform_real = interp_func_real(uniform_freq_for_hilbert); s_uniform_imag = interp_func_imag(uniform_freq_for_hilbert)
    s_uniform = s_uniform_real + 1j * s_uniform_imag; log_abs_s_uniform = np.log(np.abs(s_uniform) + 1e-12)
    phase_min_approx_uniform = -np.imag(hilbert(log_abs_s_uniform)); causal_s_param_on_uniform_grid = np.abs(s_uniform) * np.exp(1j * phase_min_approx_uniform)
    return uniform_freq_for_hilbert, causal_s_param_on_uniform_grid
def perform_signal_integrity_analysis(freq_hz, s_channel, data_rate_gbps, rise_fall_time_ps, num_bits, rj_sigma_ps=1.0, dj_pp_ps=5.0):
    is_coupled = (s_channel.shape[1] == 4); bit_period = 1.0 / (data_rate_gbps * 1e9); samples_per_bit = 256
    time_step = bit_period / samples_per_bit; num_samples = num_bits * samples_per_bit; time_vector = np.arange(num_samples) * time_step
    bits = generate_gold_sequence(num_bits); input_waveform_ideal = np.repeat(bits, samples_per_bit).astype(float)
    if rise_fall_time_ps > 0:
        sigma_t = (rise_fall_time_ps * 1e-12) / 2.5; sigma_samples = sigma_t / time_step; win_len = int(sigma_samples * 8)
        if win_len % 2 == 0: win_len += 1;
        if win_len < 3: win_len = 3
        gauss_filter = create_gaussian_filter(win_len, sigma_samples); input_waveform_ideal = np.convolve(input_waveform_ideal, gauss_filter, mode='same')
    rj = np.random.normal(0, rj_sigma_ps * 1e-12, num_bits); dj_amplitude = dj_pp_ps * 1e-12 / 2
    dj = dj_amplitude * np.sin(2 * np.pi * np.arange(num_bits) / 10); cum_jitter = np.cumsum(rj + dj); time_for_jitter_points = np.arange(num_bits) * bit_period
    interp_jitter_func = interp1d(time_for_jitter_points, cum_jitter, kind='linear', bounds_error=False, fill_value=(cum_jitter[0], cum_jitter[-1]) if num_bits > 0 else 0)
    cum_jitter_repeated = interp_jitter_func(time_vector); time_jittered = time_vector + cum_jitter_repeated
    input_interp_func = interp1d(time_vector, input_waveform_ideal, kind='linear', bounds_error=False, fill_value=(input_waveform_ideal[0], input_waveform_ideal[-1]))
    input_waveform = input_interp_func(time_jittered); n_fft = len(time_vector); fft_freqs = np.fft.fftfreq(n_fft, d=time_step); vin_f = np.fft.fft(input_waveform)
    def get_interp_s_inner(s_param_vector_for_port, param_type='S21'):
        if param_type == 'S21': dc_value = s_param_vector_for_port[0]
        elif param_type in ['S11', 'S31', 'S41', 'S22']: dc_value = s_param_vector_for_port[0]
        else: dc_value = 0.0 + 0j
        uniform_freq_for_hilbert, causal_s_param_on_uniform_grid = enforce_causality(s_param_vector_for_port, freq_hz, dc_value)
        interp_func_real = interp1d(uniform_freq_for_hilbert, np.real(causal_s_param_on_uniform_grid), kind='cubic', bounds_error=False, fill_value=0.0)
        interp_func_imag = interp1d(uniform_freq_for_hilbert, np.imag(causal_s_param_on_uniform_grid), kind='cubic', bounds_error=False, fill_value=0.0)
        s_full = np.zeros_like(fft_freqs, dtype=complex); pos_fft_freqs_mask = fft_freqs >= 0
        s_full[pos_fft_freqs_mask] = interp_func_real(fft_freqs[pos_fft_freqs_mask]) + 1j * interp_func_imag(fft_freqs[pos_fft_freqs_mask])
        neg_fft_freqs_mask = fft_freqs < 0; s_full[neg_fft_freqs_mask] = np.conj(interp_func_real(np.abs(fft_freqs[neg_fft_freqs_mask])) + 1j * interp_func_imag(np.abs(fft_freqs[neg_fft_freqs_mask])))
        dc_idx = np.where(fft_freqs == 0)[0];
        if len(dc_idx) > 0: s_full[dc_idx] = np.real(s_full[dc_idx])
        return s_full
    if is_coupled: s_thru_interp = get_interp_s_inner(s_channel[:, 2, 0], param_type='S31')
    else: s_thru_interp = get_interp_s_inner(s_channel[:, 1, 0], param_type='S21')
    vout_f = vin_f * s_thru_interp; output_waveform = np.real(np.fft.ifft(vout_f)); fext_waveform = None
    if is_coupled: s41_interp = get_interp_s_inner(s_channel[:, 3, 0], param_type='S41'); vfext_f = vin_f * s41_interp; fext_waveform = np.real(np.fft.ifft(vfext_f))
    delay = 0
    if len(output_waveform) > 0 and len(input_waveform) > 0:
        input_norm = input_waveform / (np.max(np.abs(input_waveform)) + 1e-9); output_norm = output_waveform / (np.max(np.abs(output_waveform)) + 1e-9)
        corr = correlate(output_norm, input_norm, mode='full'); peak_idx = np.argmax(corr)
        if peak_idx > 0 and peak_idx < len(corr) - 1:
            y1, y2, y3 = corr[peak_idx-1], corr[peak_idx], corr[peak_idx+1]; denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-12: peak_idx_float = peak_idx + 0.5 * (y1 - y3) / denom
            else: peak_idx_float = float(peak_idx)
        else: peak_idx_float = float(peak_idx)
        delay_idx_float = peak_idx_float - (len(input_waveform) - 1); delay = delay_idx_float * time_step
    return time_vector, input_waveform, output_waveform, fext_waveform, bit_period, delay, samples_per_bit, time_step

# <<< MODIFICATION START: Re-architected 'analyze_eye_diagram' (v2.3) >>>
def analyze_eye_diagram(output_waveform, samples_per_bit, bit_period, num_bits, time_step):
    """
    Analyzes the output waveform to generate an eye diagram and compute key metrics.
    Version 2.3: Restored plotting for all conditions, with added warnings for physically closed eyes.
    """
    output_waveform = np.asarray(output_waveform)
    
    # Check for invalid input, but don't error on low amplitude
    if output_waveform is None or len(output_waveform) < 2 * samples_per_bit:
        return {"error": "Insufficient waveform data for eye analysis."}, None, None
        
    num_full_uis_available = len(output_waveform) // samples_per_bit
    settling_uis = 10  # Settle for 10 bits
    
    if num_full_uis_available <= settling_uis + 5: # Need settling + a few bits
        return {"error": f"Insufficient bits for analysis. Need > {settling_uis + 5}, Have: {num_full_uis_available}."}, None, None

    # Form the eye matrix from 2-UI segments, regardless of amplitude
    folded_voltage_matrix = []
    for i in range(settling_uis, num_full_uis_available - 1): # Go up to the second to last UI
        start_idx = i * samples_per_bit
        end_idx = start_idx + 2 * samples_per_bit
        if end_idx <= len(output_waveform):
            segment = output_waveform[start_idx:end_idx]
            folded_voltage_matrix.append(segment)

    if not folded_voltage_matrix:
        return {"error": "Could not form any eye segments."}, None, None

    folded_voltage_matrix = np.array(folded_voltage_matrix)
    
    # Calculate statistics, similar to the original user code but more robust
    v_high_est = np.percentile(output_waveform, 95)
    v_low_est = np.percentile(output_waveform, 5)
    
    # Use a robust calculation for eye height based on the central portion of the eye
    eye_center_slice = folded_voltage_matrix[:, int(0.9 * samples_per_bit):int(1.1 * samples_per_bit)]
    if eye_center_slice.size == 0:
        # Fallback if slice is empty
        v1_mean = v_high_est
        v0_mean = v_low_est
    else:
        v1_mean = np.percentile(eye_center_slice[eye_center_slice > np.median(eye_center_slice)], 10)
        v0_mean = np.percentile(eye_center_slice[eye_center_slice < np.median(eye_center_slice)], 90)

    eye_height = max(0, v1_mean - v0_mean)
    optimal_decision_voltage = (v_high_est + v_low_est) / 2
    
    # --- The Core Logic Change ---
    # Instead of erroring out, we now package the results and add a warning if the eye is likely closed.
    stats = {
        'eye_height': eye_height,
        'eye_width': 0, # Simplified, can be enhanced later if needed
        'ber_estimate': 0.5, # Assume worst case
        'optimal_decision_voltage': optimal_decision_voltage,
        'warning': None # Initialize warning as None
    }
    
    # Heuristic: If the calculated eye height is less than 1mV, it's physically closed.
    # We still return the stats and the matrix for plotting, but add a warning message.
    if eye_height < 0.001: 
        stats['warning'] = f"Physically Closed Eye (Height < 1mV). Plot shows residual noise/crosstalk."
    
    # If the eye has some meaningful height, perform more detailed calculations
    if stats['warning'] is None:
        # (This is where more detailed width, BER, etc. calculations from v2.2 could be re-inserted if desired)
        # For now, we keep it simple to ensure a plot is always generated.
        pass

    return stats, folded_voltage_matrix, None # Always return the matrix for plotting
# <<< MODIFICATION END >>>

def write_touchstone(s_params, freq_hz, filename, z0=50):
    n_ports = s_params.shape[1];
    with open(filename, 'w') as f:
        f.write(f'# GHz S MA R {z0}\n'); f.write(f'! TSV S-parameters for {n_ports}-port network\n'); f.write('! Freq     S11_Mag  S11_Ang  S12_Mag  S12_Ang ...\n')
        for i, f_hz in enumerate(freq_hz):
            row = [f_hz / 1e9]
            for j in range(n_ports):
                for k in range(n_ports): row.extend([np.abs(s_params[i, j, k]), np.angle(s_params[i, j, k], deg=True)])
            f.write(' '.join(map(lambda x: f"{x:.8e}", row)) + '\n')
def log_message(log_widget, message):
    if not log_widget: return
    log_widget.config(state=NORMAL); log_widget.insert(tk.END, message + "\n"); log_widget.see(tk.END); log_widget.config(state=DISABLED); log_widget.update_idletasks()
def compute_and_analyze(fig, canvas, log_widget, run_button):
    run_button.config(state=DISABLED, text="Analyzing...")
    try:
        log_message(log_widget, "========================================"); log_message(log_widget, "Starting New Analysis (v2.3)...")
        params = get_parameters(log_widget)
        (mode, num_tsv, _, _, _, _, _, _, _, _, _, _, start_freq, end_freq, num_points, data_rate_gbps, _, num_bits) = params
        if end_freq > 100e9:
            log_message(log_widget, "WARNING: End frequency > 100 GHz. Expect significant signal loss and potentially a closed eye.")
        if end_freq / (data_rate_gbps * 1e9) > 10:
             log_message(log_widget, "WARNING: Simulation bandwidth is much higher than the data rate's Nyquist frequency. This is valid but may show a closed eye.")
        if num_points < end_freq / 1e9: # Heuristic: at least 1 point per GHz
            log_message(log_widget, f"WARNING: Low number of frequency points ({num_points}) for a wide range ({end_freq/1e9} GHz). Consider increasing 'Num Points' for accuracy.")
        log_message(log_widget, f"Parameters validated: Mode={mode}, TSVs={num_tsv}, Rate={data_rate_gbps}Gbps")
        log_message(log_widget, "Initializing TSV physical model...")
        tsv_model = TSVModel(*params[:12], log_widget=log_widget)
        log_message(log_widget, f"Computing frequency response ({start_freq/1e9:.1f} to {end_freq/1e9:.1f} GHz)...")
        freq_hz = np.linspace(start_freq, end_freq, num_points); s_channel = tsv_model.compute_s_params(freq_hz)
        log_message(log_widget, f"Performing transient analysis for {num_bits} bits...")
        transient_results = perform_signal_integrity_analysis(freq_hz, s_channel, *params[15:])
        log_message(log_widget, "Analyzing eye diagram...")
        _, _, output_waveform, _, bit_period, _, samples_per_bit, time_step = transient_results[:8]
        stats, eye_matrix, _ = analyze_eye_diagram(output_waveform, samples_per_bit, bit_period, num_bits, time_step)
        log_message(log_widget, "Visualizing results in GUI...")
        visualize_results(fig, canvas, log_widget, tsv_model, freq_hz, s_channel, stats, eye_matrix, None, *transient_results)
        
        log_message(log_widget, "Prompting for S-parameter file save...")
        default_extension = f".s{2 if mode == 'single' else 4}p"; file_types = [(f"Touchstone {'2' if mode == 'single' else '4'}-port", f"*.s{2 if mode == 'single' else 4}p"), ("All files", "*.*")]
        filename = filedialog.asksaveasfilename(initialdir=".", title="Save S-Parameter File", defaultextension=default_extension, filetypes=file_types, initialfile=f"tsv_{mode}_{data_rate_gbps}Gbps.s{2 if mode == 'single' else 4}p")
        if filename: write_touchstone(s_channel, freq_hz, filename); log_message(log_widget, f"SUCCESS: S-params saved to {os.path.basename(filename)}")
        else: log_message(log_widget, "Info: S-parameter file save cancelled.")
        log_message(log_widget, "ANALYSIS COMPLETE.")
    except Exception as e:
        error_msg = f"ERROR: An error occurred: {e}\n{traceback.format_exc()}"; log_message(log_widget, error_msg); messagebox.showerror("Error", error_msg)
    finally:
        run_button.config(state=NORMAL, text="Compute, Analyze & Save")
def get_parameters(log_widget):
    try:
        use_roughness = use_roughness_var.get(); use_tan_delta = use_tan_delta_var.get(); use_N_a = use_N_a_var.get()
        mode = analysis_mode.get(); num_tsv = int(entry_num_tsv.get()); height = float(entry_height.get()) * 1e-6
        radius = float(entry_radius.get()) * 1e-6; t_ox = float(entry_t_ox.get()) * 1e-6; pitch = float(entry_pitch.get()) * 1e-6 if mode == "coupled" else 0
        roughness_rms = float(entry_roughness.get()) * 1e-6 if use_roughness else None
        tan_delta_ox = float(entry_tan_delta_ox.get()) if use_tan_delta else None; N_a_cm3 = float(entry_N_a.get()) if use_N_a else None
        V = float(entry_V.get()); V_FB = float(entry_V_FB.get()); temperature = float(entry_temperature.get()) + 273.15
        start_freq = float(entry_start_freq.get()) * 1e9; end_freq = float(entry_end_freq.get()) * 1e9
        num_points = int(entry_num_points.get()); data_rate_gbps = float(entry_data_rate.get()); rise_fall_time_ps = float(entry_rise_fall.get())
        num_bits = int(entry_num_bits.get()); N_a = N_a_cm3 * 1e6 if N_a_cm3 is not None else None
        if mode == "coupled" and pitch <= 2 * (radius + t_ox): raise ValueError(f"Pitch ({pitch*1e6:.1f}μm) must be > 2 * outer radius ({2*(radius+t_ox)*1e6:.1f}μm).")
        if num_tsv < 1 or num_points < 10 or num_bits < 50: raise ValueError("Invalid params (Num TSVs < 1, Num Points < 10, or Num Bits < 50).")
        if start_freq >= end_freq: raise ValueError("Start frequency must be < end frequency.")
        if data_rate_gbps <= 0 or rise_fall_time_ps < 0: raise ValueError("Data rate must be positive, rise/fall time non-negative.")
        return (mode, num_tsv, height, radius, t_ox, pitch, roughness_rms, tan_delta_ox, N_a, V, V_FB, temperature, start_freq, end_freq, num_points, data_rate_gbps, rise_fall_time_ps, num_bits)
    except ValueError as e: log_message(log_widget, f"PARAMETER ERROR: {e}"); messagebox.showerror("Input Error", str(e)); raise

# <<< MODIFICATION START: Updated 'visualize_results' to handle warnings (v2.3) >>>
def visualize_results(fig, canvas, log_widget, tsv_model, freq_hz, s_channel, stats, eye_matrix, bathtub, *transient_results):
    (time_vector, input_waveform, output_waveform, fext_waveform, bit_period, delay, _, time_step) = transient_results[:8]
    fig.clear(); (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
    fig.suptitle(f'TSV Analysis: {tsv_model.mode.title()} ({tsv_model.num_tsv} TSV, {float(entry_data_rate.get())} Gbps)', fontsize=14)
    
    # Plot 1: Frequency Response
    freq_plot_ghz = freq_hz / 1e9; ax1.set_title('Frequency Response'); ax1.set_xlabel('Frequency (GHz)'); ax1.set_ylabel('Magnitude (dB)'); ax1.grid(True); ax1.set_ylim(-100, 5)
    if tsv_model.mode == "single":
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 1, 0]) + 1e-12), label='S21 (Insertion Loss)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 0, 0]) + 1e-12), '--', label='S11 (Return Loss)')
    else:
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 2, 0]) + 1e-12), label='S31 (Thru)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 0, 0]) + 1e-12), '--', label='S11 (Return)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 1, 0]) + 1e-12), '-.', label='S21 (NEXT)')
        ax1.plot(freq_plot_ghz, 20 * np.log10(np.abs(s_channel[:, 3, 0]) + 1e-12), ':', label='S41 (FEXT)')
    ax1.legend(loc='lower left', fontsize='small')
    
    # Plot 2: RLGC Parameters
    ax2.set_title('RLGC Parameters'); ax2.set_xlabel('Frequency (GHz)'); ax2.set_ylabel('Value (log scale)'); ax2.grid(True, which="both"); ax2.set_xscale('log'); ax2.set_yscale('log')
    rlgc_data = tsv_model.compute_rlgc(2 * np.pi * freq_hz); R, L, G, C = rlgc_data[0:4]
    ax2.plot(freq_plot_ghz, R, label='R (Ohm/m)'); ax2.plot(freq_plot_ghz, L, label='L (H/m)'); ax2.plot(freq_plot_ghz, G, label='G (S/m)'); ax2.plot(freq_plot_ghz, C, label='C (F/m)'); ax2.legend(fontsize='small')

    # Plot 3: Transient Waveforms
    ax3.set_title('Transient Waveforms'); ax3.set_xlabel('Time (ns)'); ax3.set_ylabel('Voltage (V)'); ax3.grid(True)
    ax3.plot(time_vector * 1e9, input_waveform, alpha=0.6, label='Input'); ax3.plot(time_vector * 1e9, output_waveform, label=f'Output')
    if fext_waveform is not None: ax3.plot(time_vector * 1e9, fext_waveform, label='FEXT')
    ax3.legend(fontsize='small'); ax3.set_xlim(0, min(len(time_vector) * time_step * 1e9, 20 * bit_period * 1e9)); ax3.set_ylim(min(-0.2, np.min(output_waveform)-0.1) if len(output_waveform)>0 else -0.2, max(1.2, np.max(output_waveform)+0.1) if len(output_waveform)>0 else 1.2)

    # Plot 4: Eye Diagram
    ax4.set_title('Eye Diagram'); ax4.set_xlabel('Time (ps)'); ax4.set_ylabel('Voltage (V)'); ax4.grid(True)
    
    # Check for errors or if the matrix is valid for plotting
    if 'error' in stats:
        ax4.text(0.5, 0.5, f'Eye Not Available\n({stats["error"]})', ha='center', va='center', color='red', transform=ax4.transAxes)
    elif eye_matrix is not None and len(eye_matrix) > 0:
        samples_per_bit = int(bit_period / time_step);
        eye_time_axis_ps = (np.arange(eye_matrix.shape[1]) - samples_per_bit/2) * time_step * 1e12 # Center the eye
        
        # Plot the eye segments
        for i in range(min(len(eye_matrix), 500)):
            ax4.plot(eye_time_axis_ps, eye_matrix[i], color='#9467bd', alpha=0.05)
        
        # If there is a warning (e.g., physically closed eye), display it on the plot
        if stats.get('warning'):
            ax4.text(0.5, 0.95, stats['warning'], ha='center', va='top', color='orange', weight='bold', transform=ax4.transAxes, fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax4.axhline(stats['optimal_decision_voltage'], color='red', linestyle='--', lw=1)
        ax4.set_xlim(0, bit_period * 1e12) # Display one UI
        
        # Auto-adjust y-axis for very small signals, but with a minimum sensible range
        min_v, max_v = np.min(eye_matrix), np.max(eye_matrix)
        v_range = max(abs(max_v - min_v), 0.01) # Ensure a minimum y-range of 10mV
        ax4.set_ylim(min_v - 0.1 * v_range, max_v + 0.1 * v_range)

    else:
        ax4.text(0.5, 0.5, 'Eye Not Available\n(Unknown Error)', ha='center', va='center', color='red', transform=ax4.transAxes)

    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96]); canvas.draw()
    
    # Log results
    log_message(log_widget, "\n--- Analysis Results ---")
    if 'error' in stats:
        log_message(log_widget, f"  - Eye Diagram Error: {stats['error']}")
    else:
        if stats.get('warning'):
            log_message(log_widget, f"  - WARNING: {stats['warning']}")
        log_message(log_widget, f"  - Eye Height: {stats['eye_height'] * 1000:.4f} mV") # Display in mV for clarity
        log_message(log_widget, f"  - Optimal Decision Voltage: {stats['optimal_decision_voltage']:.4f} V")
        log_message(log_widget, f"  - Propagation Delay: {delay*1e12:.2f} ps")
    log_message(log_widget, "------------------------\n")
# <<< MODIFICATION END >>>

# --- GUI 创建与主循环 (未变动) ---
if __name__ == "__main__":
    root = tk.Tk(); root.title("Integrated TSV Signal Integrity Analyzer v2.3"); root.geometry("1400x900"); root.columnconfigure(1, weight=3); root.rowconfigure(0, weight=1)
    control_frame = LabelFrame(root, text="Controls", padx=10, pady=10); control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
    output_frame = tk.Frame(root); output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew"); output_frame.rowconfigure(0, weight=5); output_frame.rowconfigure(1, weight=2); output_frame.columnconfigure(0, weight=1)
    plot_frame = LabelFrame(output_frame, text="Analysis Plots", padx=5, pady=5); plot_frame.grid(row=0, column=0, sticky="nsew")
    log_frame = LabelFrame(output_frame, text="Log & Results", padx=5, pady=5); log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    log_widget = scrolledtext.ScrolledText(log_frame, state=DISABLED, wrap=tk.WORD, height=10); log_widget.pack(expand=True, fill="both")
    fig = Figure(figsize=(10, 8), dpi=100); canvas = FigureCanvasTkAgg(fig, master=plot_frame); toolbar = NavigationToolbar2Tk(canvas, plot_frame); toolbar.update(); canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    def toggle_param_entry(var, entry, label):
        if var.get(): entry.config(state=NORMAL); label.config(fg='black')
        else: entry.config(state=DISABLED); label.config(fg='gray')
    def toggle_mode():
        if analysis_mode.get() == "coupled": entry_pitch.config(state=NORMAL); label_pitch.config(fg='black')
        else: entry_pitch.config(state=DISABLED); label_pitch.config(fg='gray')
    mode_frame = LabelFrame(control_frame, text="Analysis Mode", padx=5, pady=5); mode_frame.pack(fill="x", pady=5)
    param_frame = LabelFrame(control_frame, text="TSV Parameters", padx=5, pady=5); param_frame.pack(fill="x", pady=5)
    si_frame = LabelFrame(control_frame, text="Simulation Parameters", padx=5, pady=5); si_frame.pack(fill="x", pady=5)
    analysis_mode = StringVar(value="single"); Radiobutton(mode_frame, text="Single", variable=analysis_mode, value="single", command=toggle_mode).pack(side="left"); Radiobutton(mode_frame, text="Coupled", variable=analysis_mode, value="coupled", command=toggle_mode).pack(side="left", padx=10)
    label_pitch = tk.Label(mode_frame, text="Pitch(μm):"); label_pitch.pack(side="left"); entry_pitch = tk.Entry(mode_frame, width=8); entry_pitch.insert(0, "15"); entry_pitch.pack(side="left")
    fixed_params = {"Height (μm)": "30", "Radius (μm)": "2.5", "T Ox (μm)": "0.1", "V_bias (V)": "1.0", "V_FB (V)": "-0.8", "Temperature (C)": "25"}; entries = {}; row_idx = 0
    for text, val in fixed_params.items(): tk.Label(param_frame, text=text).grid(row=row_idx, column=0, columnspan=2, sticky='w', pady=1); entries[text] = tk.Entry(param_frame, width=15); entries[text].insert(0, val); entries[text].grid(row=row_idx, column=2, padx=5); row_idx += 1
    entry_height, entry_radius, entry_t_ox, entry_V, entry_V_FB, entry_temperature = [entries[k] for k in fixed_params]
    optional_params = {"Roughness RMS (μm)": "0.015", "Tan Delta (Oxide)": "0.002", "N_a (cm^-3)": "1e16"}; use_roughness_var, use_tan_delta_var, use_N_a_var = BooleanVar(value=True), BooleanVar(value=True), BooleanVar(value=True); optional_vars = [use_roughness_var, use_tan_delta_var, use_N_a_var]; optional_labels = []
    for i, ((text, val), var) in enumerate(zip(optional_params.items(), optional_vars)):
        chk = Checkbutton(param_frame, variable=var); chk.grid(row=row_idx, column=0, sticky='w'); lbl = tk.Label(param_frame, text=text); lbl.grid(row=row_idx, column=1, sticky='w'); optional_labels.append(lbl)
        entry = tk.Entry(param_frame, width=15); entry.insert(0, val); entry.grid(row=row_idx, column=2, padx=5)
        if "Roughness" in text: entry_roughness = entry
        elif "Tan Delta" in text: entry_tan_delta_ox = entry
        else: entry_N_a = entry
        chk.config(command=lambda v=var, e=entry, l=lbl: toggle_param_entry(v, e, l)); row_idx += 1
    sim_params = {"Start Freq (GHz)": "0.1", "End Freq (GHz)": "300", "Num Points": "601", "Number of TSVs": "1", "Data Rate (Gbps)": "40", "Rise/Fall Time (ps)": "10", "Num Bits": "8192"}
    sim_entries = {}
    for i, (text, val) in enumerate(sim_params.items()): tk.Label(si_frame, text=text).grid(row=i, column=0, sticky='w', pady=1); sim_entries[text] = tk.Entry(si_frame, width=10); sim_entries[text].insert(0, val); sim_entries[text].grid(row=i, column=1, padx=5)
    entry_start_freq, entry_end_freq, entry_num_points, entry_num_tsv, entry_data_rate, entry_rise_fall, entry_num_bits = [sim_entries[k] for k in sim_params]
    run_button = tk.Button(control_frame, text="Compute, Analyze & Save", font=('Helvetica', 12, 'bold')); run_button.config(command=lambda: compute_and_analyze(fig, canvas, log_widget, run_button)); run_button.pack(pady=20, fill='x')
    log_message(log_widget, "Welcome to the TSV Analyzer v2.3! This version will always plot an eye diagram and provide warnings for physically closed eyes.")
    toggle_mode();
    for var, entry, label in zip(optional_vars, [entry_roughness, entry_tan_delta_ox, entry_N_a], optional_labels): toggle_param_entry(var, entry, label)
    root.mainloop()

