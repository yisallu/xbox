import ctypes
from ctypes import wintypes
import numpy as np
import pyaudiowpatch as pyaudio
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue

# --- XInput 配置 ---
class XInputVibration(ctypes.Structure):
    _fields_ = [("wLeftMotorSpeed", wintypes.WORD),
                ("wRightMotorSpeed", wintypes.WORD)]

class XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD),
                ("Gamepad", wintypes.WORD * 15)]

try:
    xinput = ctypes.WinDLL("xinput1_4")
except FileNotFoundError:
    xinput = ctypes.WinDLL("xinput9_1_0")

XInputSetState = xinput.XInputSetState
XInputSetState.argtypes = [wintypes.DWORD, ctypes.POINTER(XInputVibration)]
XInputSetState.restype = wintypes.DWORD

XInputGetState = xinput.XInputGetState
XInputGetState.argtypes = [wintypes.DWORD, ctypes.POINTER(XInputState)]
XInputGetState.restype = wintypes.DWORD

XINPUT_GAMEPAD_LEFT_SHOULDER  = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200

def set_vibration(user_index, left_motor, right_motor):
    vibration = XInputVibration(int(left_motor * 65535), int(right_motor * 65535))
    return XInputSetState(user_index, ctypes.byref(vibration))

# --- 共享参数类 ---
class SharedVibrationParams:
    def __init__(self):
        self.lock = threading.Lock()
        self.vibration_enabled = True
        self.dual_motor_mode   = False
        self.vocal_only_mode   = True
        self.vibration_threshold = 300.0
        self.vibration_dead_zone = 0.05
        self.decay_rate          = 0.82
        self.map_in_max          = 5000.0
        self.gamma_correction    = 1.2
        self.current_rms  = 0.0
        self.vocal_rms    = 0.0
        self.bass_rms     = 0.0
        self.left_output  = 0.0
        self.right_output = 0.0
        self.device_name  = "未连接"

# --- 映射函数 ---
def map_value_gamma(value, in_range, out_range, gamma=1.0):
    in_min, in_max = in_range
    out_min, out_max = out_range
    value = max(in_min, min(in_max, value))
    normalized = (value - in_min) / (in_max - in_min)
    return (normalized ** gamma) * (out_max - out_min) + out_min

# --- 音频处理线程 ---
def audio_vibration_thread(shared_params, status_queue):
    FORMAT = pyaudio.paInt16
    CHUNK  = 512

    p = pyaudio.PyAudio()

    loopback_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False):
            loopback_devices.append(info)

    loopback_device = None
    try:
        wasapi_info   = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_index = wasapi_info["defaultOutputDevice"]
        default_spk   = p.get_device_info_by_index(default_index)
        status_queue.put(f"系统: 默认输出 → {default_spk['name']}")
        try:
            loopback_device = p.get_loopback_device_info_by_output_device(default_index)
        except Exception:
            for dev in loopback_devices:
                if default_spk["name"] in dev["name"]:
                    loopback_device = dev
                    break
        if loopback_device is None and loopback_devices:
            loopback_device = loopback_devices[0]
            status_queue.put("警告: 未匹配默认扬声器，使用第一个 Loopback 设备。")
    except Exception as e:
        status_queue.put(f"错误: {e}")
        p.terminate()
        return

    if loopback_device is None:
        status_queue.put("错误: 找不到任何 WASAPI Loopback 设备。")
        p.terminate()
        return

    CHANNELS = loopback_device["maxInputChannels"]
    RATE     = int(loopback_device["defaultSampleRate"])
    with shared_params.lock:
        shared_params.device_name = loopback_device["name"]

    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=loopback_device["index"],
                        frames_per_buffer=CHUNK)
        status_queue.put(f"系统: WASAPI Loopback 已启动\n  设备: {loopback_device['name']}\n  采样率: {RATE} Hz  声道: {CHANNELS}")
    except Exception as e:
        status_queue.put(f"错误: 无法打开音频流: {e}")
        p.terminate()
        return

    left_vib  = 0.0
    right_vib = 0.0
    last_btn  = False

    try:
        while True:
            if not status_queue.empty():
                try:
                    msg = status_queue.get_nowait()
                    if msg == "STOP_THREAD":
                        break
                    status_queue.put(msg)
                except queue.Empty:
                    pass

            with shared_params.lock:
                v_enabled   = shared_params.vibration_enabled
                v_dual      = shared_params.dual_motor_mode
                v_vocal     = shared_params.vocal_only_mode
                v_threshold = shared_params.vibration_threshold
                v_deadzone  = shared_params.vibration_dead_zone
                v_decay     = shared_params.decay_rate
                v_map_max   = shared_params.map_in_max
                v_gamma     = shared_params.gamma_correction

            state = XInputState()
            if XInputGetState(0, ctypes.byref(state)) == 0:
                buttons      = state.Gamepad[0]
                curr_pressed = bool((buttons & XINPUT_GAMEPAD_LEFT_SHOULDER) and
                                    (buttons & XINPUT_GAMEPAD_RIGHT_SHOULDER))
                if curr_pressed and not last_btn:
                    with shared_params.lock:
                        shared_params.vibration_enabled = not shared_params.vibration_enabled
                        v_enabled = shared_params.vibration_enabled
                    status_queue.put(f"系统: 振动 → {'开启' if v_enabled else '关闭'}")
                last_btn = curr_pressed

            try:
                data     = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(float)
                if CHANNELS > 1:
                    audio_np = audio_np.reshape(-1, CHANNELS).mean(axis=1)
            except Exception:
                continue

            n        = len(audio_np)
            fft_data = np.fft.rfft(audio_np)
            freqs    = np.fft.rfftfreq(n, 1 / RATE)

            vocal_mask = (freqs >= 600) & (freqs <= 4000)
            vocal_fft  = np.zeros_like(fft_data)
            vocal_fft[vocal_mask] = fft_data[vocal_mask]
            vocal_rms  = np.sqrt(np.mean(np.square(np.fft.irfft(vocal_fft, n=n))) + 1e-10)

            bass_mask  = (freqs >= 20) & (freqs < 600)
            bass_fft   = np.zeros_like(fft_data)
            bass_fft[bass_mask] = fft_data[bass_mask]
            bass_rms   = np.sqrt(np.mean(np.square(np.fft.irfft(bass_fft, n=n))) + 1e-10)

            total_rms  = np.sqrt(np.mean(np.square(audio_np)) + 1e-10)

            def calc_vib(rms):
                if rms > v_threshold:
                    return map_value_gamma(rms, (v_threshold, v_map_max), (0, 1), v_gamma)
                return 0.0

            if v_dual:
                target_left  = calc_vib(vocal_rms)
                target_right = calc_vib(bass_rms)
            else:
                t = calc_vib(vocal_rms if v_vocal else total_rms)
                target_left = target_right = t

            left_vib  = left_vib  * v_decay + target_left  * (1 - v_decay)
            right_vib = right_vib * v_decay + target_right * (1 - v_decay)
            if left_vib  < v_deadzone: left_vib  = 0.0
            if right_vib < v_deadzone: right_vib = 0.0

            if v_enabled:
                set_vibration(0, left_vib, right_vib)
            else:
                set_vibration(0, 0, 0)
                left_vib = right_vib = 0.0

            with shared_params.lock:
                shared_params.current_rms  = total_rms
                shared_params.vocal_rms    = vocal_rms
                shared_params.bass_rms     = bass_rms
                shared_params.left_output  = left_vib
                shared_params.right_output = right_vib

            time.sleep(0.001)

    except Exception as e:
        status_queue.put(f"运行时错误: {e}")
    finally:
        set_vibration(0, 0, 0)
        stream.close()
        p.terminate()


# ── 动态进度条组件 ──
class LabeledBar(tk.Frame):
    """带标签、数值、颜色进度条的复合控件"""
    def __init__(self, parent, label, color, max_val=5000, bar_max=1.0, mode="rms", **kwargs):
        super().__init__(parent, bg=parent["bg"] if "bg" in parent.keys() else "#f0f0f0", **kwargs)
        self.max_val  = max_val   # rms 模式的满格值
        self.bar_max  = bar_max   # vib 模式的满格值 (1.0)
        self.mode     = mode      # "rms" 或 "vib"
        self.color    = color

        self.lbl_name = tk.Label(self, text=label, width=18, anchor="w",
                                 font=("Consolas", 9), bg=self["bg"])
        self.lbl_name.pack(side="left")

        self.canvas = tk.Canvas(self, height=16, bg="#e0e0e0",
                                highlightthickness=1, highlightbackground="#bbb")
        self.canvas.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.bar_rect = self.canvas.create_rectangle(0, 0, 0, 16, fill=color, outline="")

        self.lbl_val = tk.Label(self, text="0.0", width=7, anchor="e",
                                font=("Consolas", 9, "bold"), fg=color, bg=self["bg"])
        self.lbl_val.pack(side="left")

    def set(self, value):
        self.update_idletasks()
        w = self.canvas.winfo_width()
        if w <= 1:
            return
        if self.mode == "rms":
            ratio = min(value / self.max_val, 1.0)
            text  = f"{value:.1f}"
        else:
            ratio = min(value / self.bar_max, 1.0)
            text  = f"{value:.3f}"
        self.canvas.coords(self.bar_rect, 0, 0, int(w * ratio), 16)
        self.lbl_val.config(text=text)


# --- GUI ---
class VibrationApp:
    def __init__(self, master):
        self.master = master
        master.title("Xbox Galgame 语音震动器 Pro")
        master.geometry("560x780")
        master.resizable(True, True)
        self.shared_params = SharedVibrationParams()
        self.status_queue  = queue.Queue()
        self.create_widgets()
        self.audio_thread = threading.Thread(
            target=audio_vibration_thread,
            args=(self.shared_params, self.status_queue),
            daemon=True
        )
        self.audio_thread.start()
        self.update_gui()

    def create_widgets(self):
        PAD = dict(padx=10, pady=4)

        # 设备信息
        dev_frame = ttk.LabelFrame(self.master, text="音频源 (WASAPI Loopback，无需虚拟声卡)", padding="6")
        dev_frame.pack(fill="x", **PAD)
        self.lbl_device = ttk.Label(dev_frame, text="设备: 初始化中...", foreground="blue")
        self.lbl_device.pack(anchor="w")
        ttk.Label(dev_frame, text="✅ 自动捕获系统扬声器输出", foreground="green").pack(anchor="w")

        # 马达模式
        mode_frame = ttk.LabelFrame(self.master, text="马达模式", padding="8")
        mode_frame.pack(fill="x", **PAD)

        self.dual_var = tk.BooleanVar(value=False)
        self.chk_dual = ttk.Checkbutton(
            mode_frame,
            text="🎮 双马达分离  [左马达→人声 600-4000Hz | 右马达→低频 20-600Hz]",
            variable=self.dual_var, command=self.update_dual_mode)
        self.chk_dual.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        self.vocal_var = tk.BooleanVar(value=True)
        self.chk_vocal = ttk.Checkbutton(
            mode_frame,
            text="    └─ 单马达：开启人声隔离 (过滤BGM)",
            variable=self.vocal_var, command=self.update_vocal_mode)
        self.chk_vocal.grid(row=1, column=0, columnspan=3, sticky="w", pady=1)

        # 核心参数
        param_frame = ttk.LabelFrame(self.master, text="核心参数", padding="8")
        param_frame.pack(fill="x", **PAD)

        rows = [
            ("触发阈值",   "s_thresh", "l_thresh", 0,    3000, self.update_thresh, 300.0,  "{:.1f}"),
            ("震动细腻度", "s_gamma",  "l_gamma",  0.5,  3.0,  self.update_gamma,  1.2,    "{:.2f}"),
            ("震动柔和度", "s_decay",  "l_decay",  0.5,  0.99, self.update_decay,  0.82,   "{:.2f}"),
        ]
        for i, (lbl, sattr, lattr, frm, to, cmd, init, fmt) in enumerate(rows):
            ttk.Label(param_frame, text=lbl+":").grid(row=i, column=0, sticky="w")
            lv = ttk.Label(param_frame, text=fmt.format(init), width=6)
            lv.grid(row=i, column=2)
            setattr(self, lattr, lv)
            sc = ttk.Scale(param_frame, from_=frm, to=to, orient="horizontal", command=cmd)
            sc.set(init)
            sc.grid(row=i, column=1, sticky="ew", padx=6)
            setattr(self, sattr, sc)
        param_frame.columnconfigure(1, weight=1)

        # 总开关
        self.btn_toggle = ttk.Button(self.master, text="⏸ 暂停震动", command=self.toggle_vibration)
        self.btn_toggle.pack(fill="x", **PAD)

        # ── 实时监控（进度条区域）──
        mon_frame = ttk.LabelFrame(self.master, text="实时监控", padding="10")
        mon_frame.pack(fill="x", **PAD)

        bg = mon_frame["background"] if "background" in mon_frame.keys() else "#f0f0f0"

        self.bar_total = LabeledBar(mon_frame, "原始总音量",          "#555555", max_val=5000, mode="rms")
        self.bar_total.pack(fill="x", pady=3)

        self.bar_vocal = LabeledBar(mon_frame, "人声 (600-4000Hz)",   "#27ae60", max_val=5000, mode="rms")
        self.bar_vocal.pack(fill="x", pady=3)

        self.bar_bass  = LabeledBar(mon_frame, "低频/BGM (20-600Hz)", "#2980b9", max_val=5000, mode="rms")
        self.bar_bass.pack(fill="x", pady=3)

        ttk.Separator(mon_frame, orient="horizontal").pack(fill="x", pady=6)

        self.bar_left  = LabeledBar(mon_frame, "左马达输出 (人声)",   "#27ae60", bar_max=1.0, mode="vib")
        self.bar_left.pack(fill="x", pady=3)

        self.bar_right = LabeledBar(mon_frame, "右马达输出 (低频)",   "#2980b9", bar_max=1.0, mode="vib")
        self.bar_right.pack(fill="x", pady=3)

        # 日志（只保留系统信息，不打印扬声器列表）
        self.log_text = scrolledtext.ScrolledText(self.master, height=8, state='disabled')
        self.log_text.pack(fill="both", expand=True, **PAD)

    # ── 控件回调 ──
    def update_dual_mode(self):
        dual = self.dual_var.get()
        with self.shared_params.lock:
            self.shared_params.dual_motor_mode = dual
        self.chk_vocal.config(state="disabled" if dual else "normal")

    def toggle_vibration(self):
        with self.shared_params.lock:
            self.shared_params.vibration_enabled = not self.shared_params.vibration_enabled
            enabled = self.shared_params.vibration_enabled
        self.btn_toggle.config(text="⏸ 暂停震动" if enabled else "▶ 恢复震动")

    def update_vocal_mode(self):
        with self.shared_params.lock:
            self.shared_params.vocal_only_mode = self.vocal_var.get()

    def update_thresh(self, v):
        try:
            val = float(v)
            with self.shared_params.lock: self.shared_params.vibration_threshold = val
            self.l_thresh.config(text=f"{val:.1f}")
        except AttributeError: pass

    def update_gamma(self, v):
        try:
            val = float(v)
            with self.shared_params.lock: self.shared_params.gamma_correction = val
            self.l_gamma.config(text=f"{val:.2f}")
        except AttributeError: pass

    def update_decay(self, v):
        try:
            val = float(v)
            with self.shared_params.lock: self.shared_params.decay_rate = val
            self.l_decay.config(text=f"{val:.2f}")
        except AttributeError: pass

    def update_gui(self):
        with self.shared_params.lock:
            total_rms  = self.shared_params.current_rms
            vocal_rms  = self.shared_params.vocal_rms
            bass_rms   = self.shared_params.bass_rms
            left_out   = self.shared_params.left_output
            right_out  = self.shared_params.right_output
            dev_name   = self.shared_params.device_name

        self.lbl_device.config(text=f"设备: {dev_name}")
        self.bar_total.set(total_rms)
        self.bar_vocal.set(vocal_rms)
        self.bar_bass.set(bass_rms)
        self.bar_left.set(left_out)
        self.bar_right.set(right_out)

        while not self.status_queue.empty():
            msg = self.status_queue.get_nowait()
            # 过滤掉扬声器设备列表，只显示系统状态信息
            if "可用 WASAPI" not in msg and "索引" not in msg:
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state='disabled')

        self.master.after(50, self.update_gui)

    def on_closing(self):
        self.status_queue.put("STOP_THREAD")
        set_vibration(0, 0, 0)
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
