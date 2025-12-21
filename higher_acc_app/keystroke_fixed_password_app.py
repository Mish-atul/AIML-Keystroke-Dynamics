import tkinter as tk
from tkinter import ttk, messagebox
import time
import os
import json
from collections import Counter

import numpy as np
import joblib

# -------------------------------
# CONFIG
# -------------------------------
USERS_DIR = "users"
RF_MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"  # optional

FIXED_PASSWORD = ".tie5Roanl"   # CMU benchmark password

os.makedirs(USERS_DIR, exist_ok=True)


# -------------------------------
# FEATURE EXTRACTION (password-agnostic)
# -------------------------------
def extract_features_from_events(events):
    """
    Generic feature extractor for any password (we still fix the password string,
    but timing extraction does not depend on key labels).

    events: list of dicts: {"key": str, "event": "down"/"up", "time": float}

    We compute:
    - dwell_times: key up - key down
    - flight_times: next down - current down
    - total_duration

    Then build a 47-dim vector:
    - 31 "raw" timing features (20 dwell + 10 flight + total duration)
    - 16 statistical features
    """
    if not events:
        return np.zeros(47, dtype=float)

    down_times = []
    up_times = []

    for ev in events:
        if ev["event"] == "down":
            down_times.append(ev["time"])
        elif ev["event"] == "up":
            up_times.append(ev["time"])

    down_times = np.array(down_times, dtype=float)
    up_times = np.array(up_times, dtype=float)

    n_keys = min(len(down_times), len(up_times))
    if n_keys == 0:
        return np.zeros(47, dtype=float)

    down_times = down_times[:n_keys]
    up_times = up_times[:n_keys]

    dwell_times = np.clip(up_times - down_times, 0, None)

    if n_keys > 1:
        flight_times = np.clip(down_times[1:] - down_times[:-1], 0, None)
    else:
        flight_times = np.array([], dtype=float)

    total_duration = float(up_times[-1] - down_times[0])

    # --- Raw features: 20 dwell + 10 flight + total_duration = 31 ---
    raw_feats = []

    max_dwell = 20
    dwell_padded = np.zeros(max_dwell, dtype=float)
    d_len = min(len(dwell_times), max_dwell)
    dwell_padded[:d_len] = dwell_times[:d_len]
    raw_feats.extend(dwell_padded.tolist())

    max_flight = 10
    flight_padded = np.zeros(max_flight, dtype=float)
    f_len = min(len(flight_times), max_flight)
    flight_padded[:f_len] = flight_times[:f_len]
    raw_feats.extend(flight_padded.tolist())

    raw_feats.append(total_duration)

    raw_feats = np.array(raw_feats, dtype=float)  # 31

    # --- Statistical features (16) ---
    stats = []

    def safe_stats(arr):
        if arr.size == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return (float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.percentile(arr, 90)))

    mean_d, std_d, min_d, max_d, p90_d = safe_stats(dwell_times)
    mean_f, std_f, min_f, max_f, p90_f = safe_stats(flight_times)

    stats.extend([mean_d, std_d, min_d, max_d, p90_d])
    stats.extend([mean_f, std_f, min_f, max_f, p90_f])

    stats.append(float(n_keys))               # 11
    stats.append(total_duration)              # 12

    if dwell_times.size > 1:
        stats.append(float(np.mean(np.abs(np.diff(dwell_times)))))  # 13
    else:
        stats.append(0.0)

    if flight_times.size > 0:
        stats.append(float(np.median(flight_times)))  # 14
    else:
        stats.append(0.0)

    if mean_f > 0:
        stats.append(float(mean_d / mean_f))  # 15
    else:
        stats.append(0.0)

    stats.append(float(mean_d + mean_f))      # 16

    stats = np.array(stats, dtype=float)

    full_vec = np.concatenate([raw_feats, stats], axis=0)
    assert full_vec.shape[0] == 47
    return full_vec


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


# -------------------------------
# MAIN APP CLASS
# -------------------------------
class KeystrokeAuthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keystroke Dynamics Auth (Fixed Password .tie5Roanl)")
        self.geometry("750x520")

        # Load models
        try:
            self.rf_model = joblib.load(RF_MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading RF model/scaler:\n{e}")
            self.destroy()
            return

        self.label_encoder = None
        if os.path.exists(LABEL_ENCODER_PATH):
            try:
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            except Exception:
                self.label_encoder = None

        self.reset_capture_state()
        self.create_widgets()

    # ------------------- STATE -------------------
    def reset_capture_state(self):
        self.capturing = False
        self.capture_events = []
        self.capture_start_time = None

        # Registration-specific
        self.reg_username = None
        self.reg_attempts = 0
        self.reg_features = []
        self.reg_rf_preds = []

        # Login-specific
        self.current_login_user_data = None

    # ------------------- UI -------------------
    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.reg_frame = tk.Frame(notebook)
        self.login_frame = tk.Frame(notebook)

        notebook.add(self.reg_frame, text="Registration")
        notebook.add(self.login_frame, text="Login")

        # -------- Registration Tab --------
        tk.Label(self.reg_frame, text="User Registration", font=("Arial", 16, "bold")).pack(pady=10)

        info_pw = tk.Label(
            self.reg_frame,
            text=f"Fixed password (for ALL users): {FIXED_PASSWORD}",
            font=("Arial", 12, "italic"),
            fg="blue"
        )
        info_pw.pack(pady=5)

        reg_form = tk.Frame(self.reg_frame)
        reg_form.pack(pady=10)

        tk.Label(reg_form, text="Username:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.reg_username_entry = tk.Entry(reg_form, width=30)
        self.reg_username_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(reg_form, text="Type password here:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.reg_password_entry = tk.Entry(reg_form, width=30, show="*")
        self.reg_password_entry.grid(row=1, column=1, padx=5, pady=5)

        self.reg_status_label = tk.Label(self.reg_frame, text="Attempts: 0/5", font=("Arial", 12))
        self.reg_status_label.pack(pady=5)

        btn_frame = tk.Frame(self.reg_frame)
        btn_frame.pack(pady=5)

        self.btn_start_reg = tk.Button(
            btn_frame, text="Start Registration", command=self.start_registration
        )
        self.btn_start_reg.grid(row=0, column=0, padx=5)

        self.btn_finish_attempt = tk.Button(
            btn_frame, text="Finish Attempt", command=self.finish_registration_attempt, state="disabled"
        )
        self.btn_finish_attempt.grid(row=0, column=1, padx=5)

        self.reg_info_label = tk.Label(
            self.reg_frame,
            text=(
                "Instructions:\n"
                "1. Enter a username.\n"
                f"2. Click 'Start Registration'.\n"
                f"3. Type the fixed password exactly: {FIXED_PASSWORD}\n"
                "4. Click 'Finish Attempt'. The field auto-clears.\n"
                "5. Repeat until Attempts: 5/5.\n"
                "If you mistype the password, registration resets."
            ),
            justify="left",
        )
        self.reg_info_label.pack(pady=10)

        # Bind key events for registration password box
        self.reg_password_entry.bind("<KeyPress>", self.on_key_press_reg)
        self.reg_password_entry.bind("<KeyRelease>", self.on_key_release_reg)

        # -------- Login Tab --------
        tk.Label(self.login_frame, text="User Login", font=("Arial", 16, "bold")).pack(pady=10)

        info_pw_login = tk.Label(
            self.login_frame,
            text=f"Password for login is also fixed: {FIXED_PASSWORD}",
            font=("Arial", 12, "italic"),
            fg="blue"
        )
        info_pw_login.pack(pady=5)

        login_form = tk.Frame(self.login_frame)
        login_form.pack(pady=10)

        tk.Label(login_form, text="Username:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.login_username_entry = tk.Entry(login_form, width=30)
        self.login_username_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(login_form, text="Type password here:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.login_password_entry = tk.Entry(login_form, width=30, show="*")
        self.login_password_entry.grid(row=1, column=1, padx=5, pady=5)

        self.login_status_label = tk.Label(
            self.login_frame, text="Status: Waiting for login attempt.", font=("Arial", 12)
        )
        self.login_status_label.pack(pady=5)

        login_btn_frame = tk.Frame(self.login_frame)
        login_btn_frame.pack(pady=5)

        self.btn_start_login = tk.Button(
            login_btn_frame, text="Start Login Attempt", command=self.start_login_attempt
        )
        self.btn_start_login.grid(row=0, column=0, padx=5)

        self.btn_finish_login = tk.Button(
            login_btn_frame, text="Finish Login Attempt", command=self.finish_login_attempt, state="disabled"
        )
        self.btn_finish_login.grid(row=0, column=1, padx=5)

        self.login_info_label = tk.Label(
            self.login_frame,
            text=(
                "Instructions:\n"
                "1. Enter username.\n"
                "2. Click 'Start Login Attempt'.\n"
                f"3. Type password: {FIXED_PASSWORD}\n"
                "4. Click 'Finish Login Attempt' to verify."
            ),
            justify="left",
        )
        self.login_info_label.pack(pady=10)

        # Bind key events for login password box
        self.login_password_entry.bind("<KeyPress>", self.on_key_press_login)
        self.login_password_entry.bind("<KeyRelease>", self.on_key_release_login)

    # ------------------- CAPTURE HELPERS -------------------
    def start_capture(self):
        self.capturing = True
        self.capture_events = []
        self.capture_start_time = time.time()

    def stop_capture(self):
        self.capturing = False

    def record_event(self, key, event_type):
        if not self.capturing or self.capture_start_time is None:
            return
        t = time.time() - self.capture_start_time
        self.capture_events.append({"key": key, "event": event_type, "time": t})

    # ------------------- REGISTRATION LOGIC -------------------
    def start_registration(self):
        username = self.reg_username_entry.get().strip()
        if not username:
            messagebox.showwarning("Warning", "Please enter a username.")
            return

        user_path = os.path.join(USERS_DIR, f"{username}.json")
        if os.path.exists(user_path):
            if not messagebox.askyesno("User Exists",
                                       "User already exists. Overwrite registration?"):
                return

        # Reset registration state
        self.reset_capture_state()
        self.reg_username = username
        self.reg_attempts = 0
        self.reg_features = []
        self.reg_rf_preds = []
        self.reg_status_label.config(text="Attempts: 0/5")

        self.btn_start_reg.config(state="disabled")
        self.btn_finish_attempt.config(state="normal")

        # Clear and start capture
        self.reg_password_entry.delete(0, tk.END)
        self.start_capture()
        messagebox.showinfo(
            "Registration",
            f"Registration started.\nType password: {FIXED_PASSWORD}\nThen click 'Finish Attempt'."
        )

    def finish_registration_attempt(self):
        self.stop_capture()
        typed_pw = self.reg_password_entry.get()
        self.reg_password_entry.delete(0, tk.END)

        # Check fixed password match
        if typed_pw != FIXED_PASSWORD:
            messagebox.showerror(
                "Error",
                f"Incorrect password. Expected: {FIXED_PASSWORD}\n"
                "Restarting registration."
            )
            self.btn_start_reg.config(state="normal")
            self.btn_finish_attempt.config(state="disabled")
            self.reg_status_label.config(text="Attempts: 0/5")
            self.reset_capture_state()
            return

        # Extract features
        features = extract_features_from_events(self.capture_events)
        features = features.reshape(1, -1)

        # Scale
        try:
            scaled = self.scaler.transform(features)
        except Exception as e:
            messagebox.showerror("Error", f"Error scaling features:\n{e}")
            self.btn_start_reg.config(state="normal")
            self.btn_finish_attempt.config(state="disabled")
            self.reset_capture_state()
            return

        # RF prediction (style label)
        try:
            pred = self.rf_model.predict(scaled)[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error in RF prediction:\n{e}")
            self.btn_start_reg.config(state="normal")
            self.btn_finish_attempt.config(state="disabled")
            self.reset_capture_state()
            return

        self.reg_features.append(scaled.flatten())
        self.reg_rf_preds.append(pred)
        self.reg_attempts += 1
        self.reg_status_label.config(text=f"Attempts: {self.reg_attempts}/5")

        if self.reg_attempts < 5:
            # prepare for next attempt
            self.capture_events = []
            self.capture_start_time = time.time()
            self.start_capture()
        else:
            self.complete_registration()

    def complete_registration(self):
        if len(self.reg_features) < 5:
            messagebox.showerror("Error", "Not enough attempts recorded.")
            self.btn_start_reg.config(state="normal")
            self.btn_finish_attempt.config(state="disabled")
            self.reset_capture_state()
            return

        feats = np.vstack(self.reg_features)  # (5, 47)
        mean_vec = np.mean(feats, axis=0)

        dists = [euclidean_distance(f, mean_vec) for f in feats]
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))
        threshold = mean_dist + 2 * std_dist

        if len(self.reg_rf_preds) > 0:
            most_common_rf = Counter(self.reg_rf_preds).most_common(1)[0][0]
        else:
            most_common_rf = None

        if self.label_encoder is not None and most_common_rf is not None:
            try:
                rf_label_name = self.label_encoder.inverse_transform([most_common_rf])[0]
            except Exception:
                rf_label_name = str(most_common_rf)
        else:
            rf_label_name = str(most_common_rf)

        user_data = {
            "username": self.reg_username,
            "mean_vector": mean_vec.tolist(),
            "threshold": float(threshold),
            "rf_label": int(most_common_rf) if most_common_rf is not None else None,
            "rf_label_name": rf_label_name,
        }

        user_path = os.path.join(USERS_DIR, f"{self.reg_username}.json")
        try:
            with open(user_path, "w") as f:
                json.dump(user_data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Error saving user data:\n{e}")
            self.btn_start_reg.config(state="normal")
            self.btn_finish_attempt.config(state="disabled")
            self.reset_capture_state()
            return

        messagebox.showinfo(
            "Registration Complete",
            f"User '{self.reg_username}' registered.\n"
            f"RF style label: {rf_label_name}\n"
            f"Threshold: {threshold:.4f}"
        )

        self.btn_start_reg.config(state="normal")
        self.btn_finish_attempt.config(state="disabled")
        self.reg_status_label.config(text="Attempts: 0/5")
        self.reset_capture_state()

    # ------------------- LOGIN LOGIC -------------------
    def start_login_attempt(self):
        username = self.login_username_entry.get().strip()
        if not username:
            messagebox.showwarning("Warning", "Please enter a username.")
            return

        user_path = os.path.join(USERS_DIR, f"{username}.json")
        if not os.path.exists(user_path):
            messagebox.showerror("Error", "User not found. Please register first.")
            return

        with open(user_path, "r") as f:
            self.current_login_user_data = json.load(f)

        self.login_status_label.config(
            text="Status: Capturing keystrokes... type the password and click 'Finish Login Attempt'."
        )
        self.btn_start_login.config(state="disabled")
        self.btn_finish_login.config(state="normal")

        self.login_password_entry.delete(0, tk.END)
        self.start_capture()

    def finish_login_attempt(self):
        self.stop_capture()
        typed_pw = self.login_password_entry.get()
        self.login_password_entry.delete(0, tk.END)

        if self.current_login_user_data is None:
            messagebox.showerror("Error", "No login attempt in progress.")
            self.btn_start_login.config(state="normal")
            self.btn_finish_login.config(state="disabled")
            self.login_status_label.config(text="Status: Waiting for login attempt.")
            self.reset_capture_state()
            return

        # Check fixed password
        if typed_pw != FIXED_PASSWORD:
            messagebox.showerror(
                "Error",
                f"Incorrect password during login.\nExpected: {FIXED_PASSWORD}"
            )
            self.btn_start_login.config(state="normal")
            self.btn_finish_login.config(state="disabled")
            self.login_status_label.config(text="Status: Waiting for login attempt.")
            self.reset_capture_state()
            return

        features = extract_features_from_events(self.capture_events)
        features = features.reshape(1, -1)

        try:
            scaled = self.scaler.transform(features)
        except Exception as e:
            messagebox.showerror("Error", f"Error scaling features:\n{e}")
            self.btn_start_login.config(state="normal")
            self.btn_finish_login.config(state="disabled")
            self.login_status_label.config(text="Status: Waiting for login attempt.")
            self.reset_capture_state()
            return

        mean_vec = np.array(self.current_login_user_data["mean_vector"], dtype=float)
        threshold = float(self.current_login_user_data["threshold"])
        dist = euclidean_distance(scaled.flatten(), mean_vec)

        rf_label = self.current_login_user_data.get("rf_label", None)
        rf_label_name = self.current_login_user_data.get("rf_label_name", str(rf_label))

        try:
            pred = self.rf_model.predict(scaled)[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error in RF prediction:\n{e}")
            self.btn_start_login.config(state="normal")
            self.btn_finish_login.config(state="disabled")
            self.login_status_label.config(text="Status: Waiting for login attempt.")
            self.reset_capture_state()
            return

        rf_match = (rf_label is None) or (int(pred) == int(rf_label))
        accepted = (dist < threshold) and rf_match

        msg = (
            f"Distance to template: {dist:.4f}\n"
            f"Threshold: {threshold:.4f}\n"
            f"Stored RF label: {rf_label_name}\n"
            f"Predicted RF label: {pred}"
        )

        if accepted:
            messagebox.showinfo("Login Success", "User authenticated successfully.\n\n" + msg)
            self.login_status_label.config(text="Status: Login successful.")
        else:
            messagebox.showerror("Login Failed", "Authentication failed.\n\n" + msg)
            self.login_status_label.config(text="Status: Login failed.")

        self.btn_start_login.config(state="normal")
        self.btn_finish_login.config(state="disabled")
        self.current_login_user_data = None
        self.reset_capture_state()

    # ------------------- EVENT BINDINGS -------------------
    def on_key_press_reg(self, event):
        self.record_event(getattr(event, "keysym", ""), "down")

    def on_key_release_reg(self, event):
        self.record_event(getattr(event, "keysym", ""), "up")

    def on_key_press_login(self, event):
        self.record_event(getattr(event, "keysym", ""), "down")

    def on_key_release_login(self, event):
        self.record_event(getattr(event, "keysym", ""), "up")


if __name__ == "__main__":
    app = KeystrokeAuthApp()
    app.mainloop()
