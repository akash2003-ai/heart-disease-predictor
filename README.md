# Heart Disease Predictor — Designer Frontend + Local Flask backend

This small project provides a designer UI (`index.html`) that posts patient data to a local Flask endpoint which loads the trained model and scaler saved by your notebook.

Files added:
- `index.html` — designer webpage (form + JS) that POSTs JSON to `http://localhost:5000/predict`.
- `styles.css` — page styling.
- `app.py` — small Flask server that loads `models/disease_model.pkl` and `models/scaler.pkl` and exposes `/predict`.
- `requirements.txt` — Python dependencies.

Before you start
1. Make sure the trained model and scaler from your notebook are available at:

   ```text
   c:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\disease_model.pkl
   c:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\scaler.pkl
   ```

Run steps (Windows PowerShell):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the Flask server:

```powershell
python app.py
```

4. Open `index.html` in your browser (double-click the file in Explorer or open the file URL).

Notes
- The frontend expects the server at `http://localhost:5000/predict`. If you run Flask on a different host/port, update the fetch URL inside `index.html` accordingly.
- If model/scaler files are missing, `app.py` will return an error explaining where it looked for them.

Security
- This server is for local development only. Do not expose it to untrusted networks without adding authentication and HTTPS.

Troubleshooting
- If you get `ImportError` for `flask_cors`, run `pip install flask-cors`.
- If you see `Model files not found`, verify the `models` folder and the two `.pkl` filenames.
