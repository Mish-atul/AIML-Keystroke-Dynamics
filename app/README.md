# Keystroke Dynamics Authentication System

This is a desktop GUI application that uses **keystroke dynamics** (typing patterns) combined with **Random Forest classification** to authenticate users based on their unique typing behavior.

## 📂 Project Structure

Your project folder should look like this:

```
keystroke/app/
├── keystroke_auth_app.py    # Main application (created ✓)
├── users/                     # User profiles directory (created ✓)
├── rf_model.pkl              # ⚠️ YOU NEED TO ADD THIS
├── scaler.pkl                # ⚠️ YOU NEED TO ADD THIS
└── label_encoder.pkl         # Optional (if you have it)
```

## 🔧 Where to Place Your Model Files

### **CRITICAL: Place these files in the SAME folder as `keystroke_auth_app.py`**

You need to copy your trained model files to:
```
c:\Users\ASUS\OneDrive\Desktop\keystroke\app\
```

### Required Files:
1. **`rf_model.pkl`** - Your trained Random Forest model from CMU dataset
2. **`scaler.pkl`** - The StandardScaler used during training

### Optional File:
3. **`label_encoder.pkl`** - Label encoder (for displaying user style names)

### How to Copy:
Simply copy your `.pkl` files from wherever you trained them into the `app` folder:
```
c:\Users\ASUS\OneDrive\Desktop\keystroke\app\rf_model.pkl
c:\Users\ASUS\OneDrive\Desktop\keystroke\app\scaler.pkl
c:\Users\ASUS\OneDrive\Desktop\keystroke\app\label_encoder.pkl (optional)
```

## 📦 Installation

### Prerequisites:
Make sure you have Python installed with the following packages:

```powershell
pip install numpy joblib
```

**Note:** `tkinter` comes pre-installed with standard Python distributions.

## 🚀 How to Run

1. **Navigate to the project folder:**
   ```powershell
   cd c:\Users\ASUS\OneDrive\Desktop\keystroke\app
   ```

2. **Run the application:**
   ```powershell
   python keystroke_auth_app.py
   ```

## 📖 How to Use

### **Registration Process**

1. Go to the **Registration** tab
2. Enter a **username** and **password**
3. Click **"Start Registration"**
4. Type the SAME password in the password box
5. Click **"Finish Attempt"**
6. The password field will auto-clear
7. Repeat steps 4-5 until you see **"Attempts: 5/5"**
8. You'll see a popup confirming successful registration

**Important:** 
- If you type the wrong password during any attempt, registration resets
- You must type the exact same password all 5 times

### **Login Process**

1. Go to the **Login** tab
2. Enter your **username** and **password**
3. Click **"Start Login Attempt"**
4. Type your password once
5. Click **"Finish Login Attempt"**
6. The system will verify your identity using:
   - Password correctness
   - Keystroke template matching (distance < threshold)
   - Random Forest style consistency

## 🧠 How It Works

### During Registration:
- Captures keystroke timing data for 5 password attempts
- Extracts a 47-dimensional feature vector from each attempt:
  - Dwell times (how long keys are held)
  - Flight times (time between key presses)
  - Statistical features (mean, std, min, max, percentiles)
- Scales features using your `scaler.pkl`
- Predicts typing style using your `rf_model.pkl`
- Creates a user template with:
  - Mean feature vector
  - Acceptance threshold (mean + 2*std of distances)
  - Most common RF predicted label
- Saves to `users/<username>.json`

### During Login:
- Captures keystroke timing for one password attempt
- Extracts and scales features
- Computes distance to stored template
- Predicts RF label
- **Accepts login only if:**
  - Password is correct **AND**
  - Distance < threshold **AND**
  - RF predicted label matches registered label

## 📁 User Data Storage

User profiles are stored in JSON format in the `users/` folder:

```json
{
  "username": "john_doe",
  "password": "mypassword123",
  "mean_vector": [0.123, 0.456, ...],  // 47 features
  "threshold": 2.5431,
  "rf_label": 5,
  "rf_label_name": "s005"
}
```

## 🎯 Features

- ✅ Works with **any password** (not hardcoded)
- ✅ Auto-clears password field after each attempt
- ✅ Shows attempt counter (1/5, 2/5, ...)
- ✅ Resets if wrong password is typed mid-registration
- ✅ Hybrid authentication (template + RF model)
- ✅ User-friendly Tkinter GUI
- ✅ Detailed authentication feedback

## ⚠️ Troubleshooting

### Error: "Error loading RF model/scaler"
**Solution:** Make sure `rf_model.pkl` and `scaler.pkl` are in the same folder as `keystroke_auth_app.py`

### Error: "User not found"
**Solution:** Register the user first in the Registration tab

### Login keeps failing
**Possible causes:**
1. Typing pattern is too different from registration
2. Typing too fast or too slow
3. Try registering again with consistent typing speed

## 📊 Expected Model Format

Your `rf_model.pkl` should be a trained RandomForestClassifier that:
- Expects 47 input features
- Was trained on scaled data (using the same `scaler.pkl`)

Your `scaler.pkl` should be a StandardScaler fitted on your training data.

## 🔒 Security Note

This is a **prototype** for educational/demonstration purposes. In production:
- Passwords should be hashed (not stored in plaintext)
- Use more secure storage than JSON files
- Add rate limiting for login attempts
- Consider multi-factor authentication

## 📝 License

Educational/Research Use Only
