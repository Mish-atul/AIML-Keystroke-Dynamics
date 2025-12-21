# Keystroke Dynamics Authentication (Fixed Password)

This application uses keystroke dynamics to authenticate users based on their typing patterns with a **fixed password**: `.tie5Roanl`

## 📂 Required Files

Place the following files in the **same folder** as `keystroke_fixed_password_app.py`:

```
higher_acc_app/
├── keystroke_fixed_password_app.py  ✅ (Created)
├── rf_model.pkl                      ⚠️ (YOU NEED TO ADD THIS)
├── scaler.pkl                        ⚠️ (YOU NEED TO ADD THIS)
└── label_encoder.pkl                 ⚠️ (OPTIONAL - ADD IF YOU HAVE IT)
```

### Where to Add Model Files:

**Copy your model files to:**
```
c:\Users\ASUS\OneDrive\Desktop\keystroke\higher_acc_app\
```

The app expects:
- `rf_model.pkl` - Your trained Random Forest model
- `scaler.pkl` - Your feature scaler (StandardScaler or similar)
- `label_encoder.pkl` - (Optional) Label encoder for user labels

## 📦 Installation

Install required Python packages:

```bash
pip install numpy joblib
```

## 🚀 How to Run

```bash
python keystroke_fixed_password_app.py
```

## 🔐 How It Works

### Registration:
1. Go to **Registration** tab
2. Enter a **username**
3. Click **"Start Registration"**
4. Type the fixed password: `.tie5Roanl`
5. Click **"Finish Attempt"**
6. Repeat **5 times** (the app tracks: Attempts: X/5)
7. If you mistype the password, registration resets

### Login:
1. Go to **Login** tab
2. Enter your **username**
3. Click **"Start Login Attempt"**
4. Type the fixed password: `.tie5Roanl`
5. Click **"Finish Login Attempt"**
6. The app shows:
   - Distance to your typing template
   - Threshold value
   - RF style label (stored vs predicted)
   - ✅ Success or ❌ Failure

## ⏱️ Timing Details

- **Timing starts**: The moment you press the first key
- **Timing stops**: After the last key is released
- The app captures:
  - Dwell times (key down → key up)
  - Flight times (key down → next key down)
  - Total duration
  - Statistical features

## 🎯 Why Fixed Password?

Using the same password (`.tie5Roanl`) for all users:
- Matches the CMU dataset structure
- Makes feature distribution consistent with training data
- Improves RF prediction stability
- Increases template distance sharpness
- **Results in higher accuracy**

## 📊 Authentication Method

**Hybrid approach:**
- ✅ Distance to template < threshold
- ✅ RF label matches stored label

Both conditions must be true for successful authentication.

## 📁 User Data

User profiles are saved in the `users/` folder (auto-created) as JSON files:
```json
{
  "username": "alice",
  "mean_vector": [...],
  "threshold": 12.3456,
  "rf_label": 42,
  "rf_label_name": "user_042"
}
```
