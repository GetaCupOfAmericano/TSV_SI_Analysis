import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, LabelFrame

# --- 物理常量 ---
eps_0, mu_0, q = 8.85e-12, 4 * np.pi * 1e-7, 1.6e-19

# --- S参数/ABCD参数转换工具函数 (无变化) ---
def s_to_abcd(s, z0=50):
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    denom = 2 * s21 * z0
    if abs(denom) < 1e-15: return np.identity(2) * float('inf')
    A = ((1 + s11) * (1 - s22) + s12 * s21) / denom * z0
    B = ((1 + s11) * (1 + s22) - s12 * s21) / denom * z0**2
    C = ((1 - s11) * (1 - s22) - s12 * s21) / denom
    D = ((1 - s11) * (1 + s22) + s12 * s21) / denom * z0
    return np.array([[A, B], [C, D]])

def abcd_to_s(abcd, z0=50):
    A, B, C, D = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
    denom = A * z0 + B + C * z0**2 + D * z0
    if abs(denom) < 1e-15: return np.zeros((2,2), dtype=complex)
    s11 = (A * z0 + B - C * z0**2 - D * z0) / denom
    s12 = 2 * (A * D - B * C) * z0 / denom
    s21 = 2 * z0 / denom
    s22 = (-A * z0 + B - C * z0**2 + D * z0) / denom
    return np.array([[s11, s12], [s21, s22]])

def create_gaussian_filter(win_len, sigma):
    x = np.linspace(-(win_len - 1) / 2., (win_len - 1) / 2., win_len)
    gauss = np.exp(-0.5 * (x / sigma)**2)
    return gauss / np.sum(gauss)

# --- 【新增】德拜介电模型函数 ---
def calculate_debye_permittivity(f, eps_s, eps_inf, tau):
    """
    根据德拜模型计算频率f下的复数介电常数及其相关参数。
    f: 频率 (Hz)
    eps_s: 静态相对介电常数
    eps_inf: 无穷频率相对介电常数
    tau: 弛豫时间 (s)
    返回: eps_real (实际相对介电常数), tan_delta (损耗角正切)
    """
    omega = 2 * np.pi * f
    # 计算复数相对介电常数
    epsilon_complex_relative = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
    
    # 提取实部和虚部
    eps_real = epsilon_complex_relative.real
    eps_imag = -epsilon_complex_relative.imag # 按照惯例，损耗为正
    
    # 计算损耗角正切
    if abs(eps_real) < 1e-12:
        tan_delta = 0
    else:
        tan_delta = eps_imag / eps_real
        
    return eps_real, tan_delta

# 眼图分析函数 (暂时搁置，保持原样)
def analyze_eye_diagram(output_waveform, samples_per_bit, bit_period):
    """
    【最终修正版 v3】使用自定义的、鲁棒的插值方法来精确计算转换时间。
    """
    num_bits_total = len(output_waveform) // samples_per_bit
    settling_bits = 20
    if num_bits_total <= settling_bits + 3:
        return {"error": "比特数太少(<23)，无法进行精确分析。"}, None

    eye_matrix = np.array([output_waveform[i*samples_per_bit : (i+2)*samples_per_bit]
                           for i in range(settling_bits, num_bits_total - 2)])

    # 1. 精确高低电平检测 (此部分逻辑稳定，无需修改)
    eye_center_start = int(samples_per_bit * 0.9)
    eye_center_end = int(samples_per_bit * 1.1)
    voltage_samples = eye_matrix[:, eye_center_start:eye_center_end].flatten()
    hist, bin_edges = np.histogram(voltage_samples, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        smooth_hist = np.convolve(hist, np.ones(5)/5, 'same')
        valleys = np.where((smooth_hist[1:-1] < smooth_hist[:-2]) & (smooth_hist[1:-1] < smooth_hist[2:]))[0] + 1
        if not len(valleys): raise ValueError("No valley in histogram")
        mid_idx = valleys[np.argmin(smooth_hist[valleys])]
        v_thresh = bin_centers[mid_idx]
        ones = voltage_samples[voltage_samples > v_thresh]
        zeros = voltage_samples[voltage_samples < v_thresh]
        if len(ones) < 20 or len(zeros) < 20: raise ValueError("电平分离失败")
        v_high_mean, v_high_std = np.mean(ones), np.std(ones)
        v_low_mean, v_low_std = np.mean(zeros), np.std(zeros)
    except Exception:
        return {"error": "无法识别高/低电平，眼图可能已闭合。"}, None

    # 2. 计算眼高 (逻辑不变)
    eye_height = (v_high_mean - 3 * v_high_std) - (v_low_mean + 3 * v_low_std)

    # 3. 【核心修正】使用自定义插值函数计算上升/下降时间
    time_axis = np.linspace(0, 2 * bit_period, 2 * samples_per_bit)
    v10 = v_low_mean + 0.1 * (v_high_mean - v_low_mean)
    v90 = v_low_mean + 0.9 * (v_high_mean - v_low_mean)

    rising_edges, falling_edges = [], []
    center_idx = samples_per_bit
    for seg in eye_matrix:
        if np.mean(seg[center_idx-5:center_idx]) < v_thresh and np.mean(seg[center_idx:center_idx+5]) > v_thresh:
            rising_edges.append(seg)
        elif np.mean(seg[center_idx-5:center_idx]) > v_thresh and np.mean(seg[center_idx:center_idx+5]) < v_thresh:
            falling_edges.append(seg)

    def get_transition_time(edges, is_rising):
        if not edges: return 0
        avg_edge = np.mean(np.array(edges), axis=0)

        # ---- 自定义、鲁棒的线性插值函数 ----
        def manual_interp(v_target, voltages, times):
            # 找到电压第一次穿过目标阈值的点
            if is_rising:
                indices = np.where(voltages >= v_target)[0]
            else: # 下降沿，从高到低找
                indices = np.where(voltages <= v_target)[0]
            
            if not indices.size: return None # 阈值从未被达到
            
            idx = indices[0] # 第一个过阈值的点
            if idx == 0: return times[0] # 在第一个点就过阈值了

            # 获取插值所需的前后两个点
            t1, v1 = times[idx-1], voltages[idx-1]
            t2, v2 = times[idx], voltages[idx]

            # 避免除以零
            if abs(v2 - v1) < 1e-12: return t1
            
            # 标准线性插值公式
            t_cross = t1 + (t2 - t1) * (v_target - v1) / (v2 - v1)
            return t_cross
        # ---- 函数结束 ----
        
        t10 = manual_interp(v10, avg_edge, time_axis)
        t90 = manual_interp(v90, avg_edge, time_axis)

        if t10 is None or t90 is None:
            return 0 # 如果找不到对应的点，则返回0

        return abs(t90 - t10)

    rise_time = get_transition_time(rising_edges, True)
    fall_time = get_transition_time(falling_edges, False)

    # 4. 精确眼宽和抖动 (逻辑不变)
    crossing_times = []
    for seg in eye_matrix:
        cross_indices = np.where(np.diff(np.sign(seg - v_thresh)))[0]
        for idx in cross_indices:
            v1, v2, t1, t2 = seg[idx], seg[idx+1], time_axis[idx], time_axis[idx+1]
            if abs(v2 - v1) > 1e-9:
                t_cross = t1 + (v_thresh - v1) * (t2 - t1) / (v2 - v1)
                crossing_times.append(t_cross)

    center_crossings = np.array([t for t in crossing_times if 0.8*bit_period < t < 1.2*bit_period])
    if len(center_crossings) < 10:
        jitter_pp, eye_width = 0, 0
    else:
        jitter_pp = np.max(center_crossings) - np.min(center_crossings)
        eye_width = bit_period - jitter_pp

    return {
        "eye_height_V": max(0, eye_height),
        "eye_width_ps": max(0, eye_width * 1e12),
        "rise_time_ps": rise_time * 1e12,
        "fall_time_ps": fall_time * 1e12,
        "jitter_pp_ps": jitter_pp * 1e12,
    }, eye_matrix


# 主分析流程 (无变化)
def perform_signal_integrity_analysis(freq_hz, s_channel, data_rate_gbps, num_bits, rise_fall_time_ps):
    # 此函数代码无变化
    data_rate, bit_period = data_rate_gbps * 1e9, 1.0 / (data_rate_gbps * 1e9)
    rise_fall_time = rise_fall_time_ps * 1e-12
    samples_per_bit = 64
    time_step = bit_period / samples_per_bit
    sim_time = num_bits * bit_period
    time_vector = np.arange(0, sim_time, time_step)
    bits = np.random.randint(0, 2, num_bits)
    input_waveform_ideal = np.repeat(bits, samples_per_bit).astype(float)
    if rise_fall_time > 0:
        sigma_t = rise_fall_time / 2.5
        sigma_samples = sigma_t / time_step
        win_len = int(sigma_samples * 8) + 1
        if win_len > 1:
            gauss_filter = create_gaussian_filter(win_len, sigma=sigma_samples)
            input_waveform = np.convolve(input_waveform_ideal, gauss_filter, mode='same')
        else: input_waveform = input_waveform_ideal
    else: input_waveform = input_waveform_ideal
    n_fft = len(time_vector)
    vin_f = np.fft.fft(input_waveform)
    fft_freqs = np.fft.fftfreq(n_fft, d=time_step)
    s21_channel = s_channel[:, 1, 0]
    s21_abs_interp = np.interp(np.abs(fft_freqs), freq_hz, np.abs(s21_channel), left=np.abs(s21_channel[0]), right=np.abs(s21_channel[-1]))
    s21_phase_interp = np.interp(np.abs(fft_freqs), freq_hz, np.unwrap(np.angle(s21_channel)), left=np.angle(s21_channel[0]), right=np.unwrap(np.angle(s21_channel))[-1])
    s21_interp_complex = s21_abs_interp * np.exp(1j * s21_phase_interp)
    s21_full_spectrum = np.zeros_like(fft_freqs, dtype=complex)
    positive_freq_mask = (fft_freqs >= 0)
    s21_full_spectrum[positive_freq_mask] = s21_interp_complex[positive_freq_mask]
    neg_freq_indices = np.where(fft_freqs < 0)[0]
    pos_freq_map = {int(round(f)): i for i, f in enumerate(fft_freqs[positive_freq_mask])}
    for i in neg_freq_indices:
        f_neg = fft_freqs[i]
        if int(round(-f_neg)) in pos_freq_map:
            s21_full_spectrum[i] = np.conj(s21_full_spectrum[pos_freq_map[int(round(-f_neg))]])
    vout_f = vin_f * s21_full_spectrum
    output_waveform = np.real(np.fft.ifft(vout_f))
    try:
        input_50_idx = np.where(input_waveform > 0.5)[0][5*samples_per_bit]
        output_50_idx = np.where(output_waveform > np.mean(output_waveform))[0]
        output_crossing_after_input = output_50_idx[output_50_idx > input_50_idx - samples_per_bit][0]
        delay = (output_crossing_after_input - input_50_idx) * time_step
    except IndexError: delay = 0
    return time_vector, input_waveform, output_waveform, bit_period, delay, samples_per_bit

def compute_and_analyze():
    try:
        # --- GUI读取 ---
        # 【修改】读取新增参数
        height = float(entry_height.get()) * 1e-6
        radius = float(entry_radius.get()) * 1e-6
        sigma_cu = float(entry_sigma_cu.get())
        roughness_rms = float(entry_roughness.get()) * 1e-6 # RMS Roughness in meters
        
        eps_si_static = float(entry_eps_si_static.get())
        eps_si_inf = float(entry_eps_si_inf.get())
        tau_si = float(entry_tau_si.get()) * 1e-12 # Relaxation time in seconds
        
        eps_ox = float(entry_eps_ox.get())
        tan_delta_ox = float(entry_tan_delta_ox.get()) # 【修改】Tan_delta现在专指Oxide
        
        sigma_si = float(entry_sigma_si.get())
        t_ox = float(entry_t_ox.get()) * 1e-6
        N_a = float(entry_N_a.get())
        V = float(entry_V.get())
        V_FB = float(entry_V_FB.get())
        
        start_freq, end_freq, num_points = float(entry_start_freq.get()) * 1e9, float(entry_end_freq.get()) * 1e9, int(entry_num_points.get())
        if start_freq == 0: start_freq = 1e6
        
        # --- 【核心修改】计算单个TSV的S参数 (集成新模型) ---
        freq_hz = np.linspace(start_freq, end_freq, num_points)
        s_single_tsv = np.zeros((len(freq_hz), 2, 2), dtype=complex)
        omega = 2 * np.pi * freq_hz

        for i, f in enumerate(freq_hz):
            omega_i = omega[i]
            if omega_i <= 0: continue
            
            # --- 【集成点1】频率相关的介电常数计算 ---
            eps_si_real, tan_delta_si = calculate_debye_permittivity(f, eps_si_static, eps_si_inf, tau_si)

            # --- R, L, G, C 参数计算 (使用新模型) ---
            w_dep = np.sqrt(2 * eps_si_real * eps_0 * abs(V - V_FB) / (q * N_a)) if (q * N_a) > 0 else 0
            r_ox = radius + t_ox
            
            C_ox = 2 * np.pi * eps_ox * eps_0 / np.log(r_ox / radius) if r_ox > radius else 0
            C_dep = 2 * np.pi * eps_si_real * eps_0 / np.log((r_ox + w_dep) / r_ox) if (r_ox + w_dep) > r_ox else 0
            C = 1 / (1 / C_ox + 1 / C_dep) if C_ox > 0 and C_dep > 0 else 0
            
            G_dc_si = 2 * np.pi * sigma_si / np.log((r_ox + w_dep) / r_ox) if (r_ox + w_dep) > r_ox else 0
            # 总电导 G = 直流电导 + 介质损耗 (来自Si和Ox)
            G = G_dc_si + omega_i * (C_dep * tan_delta_si + C_ox * tan_delta_ox)
            
            # --- 【集成点2】导体电阻计算 (趋肤效应 + 表面粗糙度) ---
            R_dc = 1 / (np.pi * radius**2 * sigma_cu) if radius > 0 and sigma_cu > 0 else float('inf')
            delta_skin = np.sqrt(2 / (omega_i * mu_0 * sigma_cu)) if (omega_i * mu_0 * sigma_cu) > 0 else float('inf')
            
            # 基础趋肤效应电阻
            if delta_skin < radius:
                R_skin_effect = R_dc * (radius / (2 * delta_skin))
            else:
                R_skin_effect = 0 # 低频下趋肤效应不明显
            
            R_ac = R_dc + R_skin_effect

            # Hammerstad表面粗糙度修正
            if roughness_rms > 0 and delta_skin < float('inf'):
                K_rough = 1 + (2 / np.pi) * np.arctan(1.4 * (roughness_rms / delta_skin)**2)
                R = R_ac * K_rough
            else:
                R = R_ac
            
            # 电感 L (与之前类似，但可进一步精细化)
            L_ext = (mu_0 / (2 * np.pi)) * np.log(2 * height / radius + 0.5) if height > 0 and radius > 0 else 0
            L_int = (mu_0 / (2 * np.pi)) * (delta_skin / (2 * radius)) if delta_skin < radius else mu_0 / (8 * np.pi)
            L = L_ext + L_int

            # 传输线参数计算 (无变化)
            Z_series, Y_parallel = R + 1j * omega_i * L, G + 1j * omega_i * C
            gamma = np.sqrt(Z_series * Y_parallel)
            Zc = np.sqrt(Z_series / Y_parallel) if abs(Y_parallel) > 1e-12 else float('inf')
            A, D = np.cosh(gamma * height), np.cosh(gamma * height)
            B = Zc * np.sinh(gamma * height)
            C_abcd = np.sinh(gamma * height) / Zc if abs(Zc) > 1e-12 else 0
            Z0 = 50
            denom = A + B / Z0 + C_abcd * Z0 + D
            s_single_tsv[i, 0, 0] = (A + B / Z0 - C_abcd * Z0 - D) / denom
            s_single_tsv[i, 1, 0] = 2 / denom
            s_single_tsv[i, 0, 1] = s_single_tsv[i, 1, 0]
            s_single_tsv[i, 1, 1] = s_single_tsv[i, 0, 0]

        # --- 多TSV级联 和 后续分析 (无变化) ---
        num_tsv = int(entry_num_tsv.get())
        s_channel = np.zeros_like(s_single_tsv)
        if num_tsv == 1:
            s_channel = s_single_tsv
        else:
            for i in range(len(freq_hz)):
                abcd_single = s_to_abcd(s_single_tsv[i, :, :])
                abcd_total = np.linalg.matrix_power(abcd_single, num_tsv)
                s_channel[i, :, :] = abcd_to_s(abcd_total)

        data_rate_gbps, num_bits, rise_fall_time_ps = float(entry_data_rate.get()), int(entry_num_bits.get()), float(entry_rise_fall.get())
        time_vector, input_waveform, output_waveform, bit_period, delay, samples_per_bit = \
            perform_signal_integrity_analysis(freq_hz, s_channel, data_rate_gbps, num_bits, rise_fall_time_ps)
        
        eye_metrics, eye_matrix = analyze_eye_diagram(output_waveform, samples_per_bit, bit_period)

        # --- 绘图 (无变化) ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
        fig.suptitle(f'Signal Integrity Analysis for {num_tsv} Cascaded TSV(s) [Advanced Model]', fontsize=16)
        ax1.plot(freq_hz / 1e9, 20 * np.log10(np.abs(s_channel[:, 1, 0])), 'b-', label='S21 (Insertion Loss)')
        ax1.plot(freq_hz / 1e9, 20 * np.log10(np.abs(s_channel[:, 0, 0])), 'r--', label='S11 (Return Loss)')
        ax1.set_title(f'Channel Frequency Response ({num_tsv} TSVs)')
        ax1.set_xlabel('Frequency (GHz)'); ax1.set_ylabel('Magnitude (dB)'); ax1.legend(); ax1.grid(True)
        ax2.plot(time_vector * 1e9, input_waveform, 'r-', alpha=0.7, label='Input Signal')
        ax2.plot(time_vector * 1e9, output_waveform, 'b-', label=f'Output Signal (Delay ≈ {delay*1e12:.2f} ps)')
        ax2.set_title(f'Transient Analysis @ {data_rate_gbps} Gbps'); ax2.set_xlabel('Time (ns)'); ax2.set_ylabel('Voltage (V)')
        ax2.legend(); ax2.grid(True); ax2.set_xlim(20 * bit_period * 1e9, 35 * bit_period * 1e9); ax2.set_ylim(-0.2, 1.2)
        ax3.set_title('Eye Diagram'); ax3.set_xlabel(f'Time (ps)      [1 UI = {bit_period*1e12:.2f} ps]'); ax3.set_ylabel('Voltage (V)'); ax3.grid(True)
        if eye_matrix is not None:
            eye_time_axis = np.linspace(-0.5 * bit_period, 1.5 * bit_period, 2 * samples_per_bit) * 1e12
            for segment in eye_matrix: ax3.plot(eye_time_axis, segment, 'b-', alpha=0.03)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.subplots_adjust(hspace=0.45)
        plt.show()

        status_label.config(text=f"Analysis for {num_tsv} TSV(s) complete!")

    except ValueError as e: messagebox.showerror("Input Error", f"Invalid input: {e}")
    except Exception as e: messagebox.showerror("Error", f"An error occurred: {e}"); import traceback; traceback.print_exc()

# --- GUI 界面创建 (【修改】增加新参数输入框) ---
root = tk.Tk()
root.title("Advanced TSV Channel Analyzer")

param_frame = LabelFrame(root, text="TSV Physical & Material Parameters", padx=10, pady=10)
param_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

# --- 参数布局重新整理以符合逻辑 ---
# 几何参数
tk.Label(param_frame, text="Height (μm):").grid(row=0, column=0, sticky='w'); entry_height = tk.Entry(param_frame); entry_height.insert(0, "30"); entry_height.grid(row=0, column=1)
tk.Label(param_frame, text="Radius (μm):").grid(row=0, column=2, padx=(10,0), sticky='w'); entry_radius = tk.Entry(param_frame); entry_radius.insert(0, "1.5"); entry_radius.grid(row=0, column=3)
tk.Label(param_frame, text="T Ox (μm):").grid(row=0, column=4, padx=(10,0), sticky='w'); entry_t_ox = tk.Entry(param_frame); entry_t_ox.insert(0, "0.2"); entry_t_ox.grid(row=0, column=5)

# 导体(Copper)参数
tk.Label(param_frame, text="Sigma Cu (S/m):").grid(row=1, column=0, sticky='w'); entry_sigma_cu = tk.Entry(param_frame); entry_sigma_cu.insert(0, "5.8e7"); entry_sigma_cu.grid(row=1, column=1)
tk.Label(param_frame, text="Roughness RMS (μm):").grid(row=1, column=2, padx=(10,0), sticky='w'); entry_roughness = tk.Entry(param_frame); entry_roughness.insert(0, "0.15"); entry_roughness.grid(row=1, column=3) # 【新增】

# 氧化层(Oxide)参数
tk.Label(param_frame, text="Eps Ox:").grid(row=2, column=0, sticky='w'); entry_eps_ox = tk.Entry(param_frame); entry_eps_ox.insert(0, "3.9"); entry_eps_ox.grid(row=2, column=1)
tk.Label(param_frame, text="Tan Delta (Oxide):").grid(row=2, column=2, padx=(10,0), sticky='w'); entry_tan_delta_ox = tk.Entry(param_frame); entry_tan_delta_ox.insert(0, "0.001"); entry_tan_delta_ox.grid(row=2, column=3) # 【修改】

# 衬底(Silicon)参数 - 德拜模型
tk.Label(param_frame, text="Eps Si (Static):").grid(row=3, column=0, sticky='w'); entry_eps_si_static = tk.Entry(param_frame); entry_eps_si_static.insert(0, "11.7"); entry_eps_si_static.grid(row=3, column=1) # 【修改】
tk.Label(param_frame, text="Eps Si (Infinite F):").grid(row=3, column=2, padx=(10,0), sticky='w'); entry_eps_si_inf = tk.Entry(param_frame); entry_eps_si_inf.insert(0, "11.7"); entry_eps_si_inf.grid(row=3, column=3) # 【新增】
tk.Label(param_frame, text="Tau Si (ps):").grid(row=3, column=4, padx=(10,0), sticky='w'); entry_tau_si = tk.Entry(param_frame); entry_tau_si.insert(0, "10"); entry_tau_si.grid(row=3, column=5) # 【新增】
tk.Label(param_frame, text="Sigma Si (S/m):").grid(row=4, column=0, sticky='w'); entry_sigma_si = tk.Entry(param_frame); entry_sigma_si.insert(0, "10"); entry_sigma_si.grid(row=4, column=1) # 【修改】默认值
tk.Label(param_frame, text="N_a (m^-3):").grid(row=4, column=2, padx=(10,0), sticky='w'); entry_N_a = tk.Entry(param_frame); entry_N_a.insert(0, "1e21"); entry_N_a.grid(row=4, column=3)
tk.Label(param_frame, text="V_bias (V):").grid(row=5, column=0, sticky='w'); entry_V = tk.Entry(param_frame); entry_V.insert(0, "1.0"); entry_V.grid(row=5, column=1)
tk.Label(param_frame, text="V_FB (V):").grid(row=5, column=2, padx=(10,0), sticky='w'); entry_V_FB = tk.Entry(param_frame); entry_V_FB.insert(0, "0.0"); entry_V_FB.grid(row=5, column=3)

# --- 信号与仿真参数 ---
si_frame = LabelFrame(root, text="Simulation & Signal Parameters", padx=10, pady=10)
si_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
tk.Label(si_frame, text="Start Freq (GHz):").grid(row=0, column=0, sticky='w'); entry_start_freq = tk.Entry(si_frame); entry_start_freq.insert(0, "0.1"); entry_start_freq.grid(row=0, column=1)
tk.Label(si_frame, text="End Freq (GHz):").grid(row=0, column=2, padx=(10,0), sticky='w'); entry_end_freq = tk.Entry(si_frame); entry_end_freq.insert(0, "50"); entry_end_freq.grid(row=0, column=3)
tk.Label(si_frame, text="Num Points:").grid(row=0, column=4, padx=(10,0), sticky='w'); entry_num_points = tk.Entry(si_frame); entry_num_points.insert(0, "501"); entry_num_points.grid(row=0, column=5)
tk.Label(si_frame, text="Number of TSVs:").grid(row=1, column=0, sticky='w'); entry_num_tsv = tk.Entry(si_frame); entry_num_tsv.insert(0, "1"); entry_num_tsv.grid(row=1, column=1)
tk.Label(si_frame, text="Data Rate (Gbps):").grid(row=1, column=2, sticky='w', padx=(10,0)); entry_data_rate = tk.Entry(si_frame); entry_data_rate.insert(0, "20"); entry_data_rate.grid(row=1, column=3)
tk.Label(si_frame, text="Num Bits:").grid(row=1, column=4, sticky='w', padx=(10,0)); entry_num_bits = tk.Entry(si_frame); entry_num_bits.insert(0, "1000"); entry_num_bits.grid(row=1, column=5)
tk.Label(si_frame, text="Rise/Fall Time (ps):").grid(row=2, column=0, sticky='w'); entry_rise_fall = tk.Entry(si_frame); entry_rise_fall.insert(0, "10"); entry_rise_fall.grid(row=2, column=1)

tk.Button(root, text="Compute and Analyze Channel", command=compute_and_analyze).grid(row=2, column=0, pady=10)
status_label = tk.Label(root, text="Enter parameters and click 'Compute'")
status_label.grid(row=3, column=0, pady=5)

root.mainloop()

