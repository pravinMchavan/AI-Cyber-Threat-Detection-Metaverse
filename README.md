# AI-Cyber-Threat-Detection-Metaverse (MCA Final Year Project)

This is a final year MCA project.

MVP (first working version):
- Detect suspicious/anomalous network activity from a CSV file
- Show prediction results in a small demo app (Streamlit)

Extended version (final project tasks):
- Simulated metaverse server environment (Flask) that generates live telemetry
- Synthetic DDoS + phishing data generation
- Hybrid anomaly detection (supervised + unsupervised)
- Real-time monitoring dashboard with alert logging
- Evaluation of detection accuracy + false positives
- HTTPS/TLS enabled for simulator communication

Simple meaning of words:
- MVP = smallest working version you can demo
- CSV = table file (rows/columns) like Excel
- Model = trained program that predicts Normal vs Attack

## Folder structure

```
AI-Cyber-Threat-Detection-Metaverse/
	artifacts/            # outputs (trained model, metrics) [auto-created]
	docs/                 # threat study + mitigation docs
	data/
		raw/                # put your raw CSV datasets here
		processed/          # optional: cleaned/processed CSVs
	scripts/
		train_model.py      # trains the baseline model
		run_simulator.py    # runs the simulated metaverse server
		collect_synthetic_data.py  # saves labeled synthetic datasets
		evaluate_models.py  # evaluates accuracy + false positives
		test_attack_scenarios.py   # test DDoS + phishing scenarios
		run_app.py          # runs the Streamlit demo app
	src/
		config.py           # paths + constants
		data/               # load CSV, create sample synthetic dataset
		models/             # training + prediction
		app/                # Streamlit UI
```

## Setup (Windows)

1) Create and activate a virtual environment (recommended)

```powershell
cd C:\Pravin\Project\AI-Cyber-Threat-Detection-Metaverse
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Train the model

Train using sample (synthetic) data:

```powershell
py scripts\train_model.py
```

This trains:
- Supervised RandomForest (saved to `artifacts/model.joblib`)
- Unsupervised IsolationForest (saved to `artifacts/model_unsupervised.joblib`)

Train using your own CSV (must include the required columns + `label` column):

```powershell
py scripts\train_model.py --csv data\raw\your_dataset.csv
```

`label` meaning:
- `0` = Normal
- `1` = Attack

## Run the demo app

```powershell
py scripts\run_app.py
```

If you prefer, you can run Streamlit directly:

```powershell
streamlit run src\app\streamlit_app.py
```

## CSV columns expected

Your CSV should include these columns:
- `duration`, `src_bytes`, `dst_bytes`, `packets`, `protocol`, `service`, `flag`

For training, you must also include:
- `label`

## Run the simulated metaverse server (Flask)

Start simulator on HTTPS with a self-signed certificate:

```powershell
py scripts\run_simulator.py --https
```

Check endpoints:
- `https://127.0.0.1:5050/health`
- `https://127.0.0.1:5050/events?limit=20`

Note: the cert is self-signed for demo only.

## Live monitoring dashboard

1) Start the simulator (above)
2) Start Streamlit:

```powershell
py scripts\run_app.py
```

3) Open the **Live Monitoring** tab
4) Keep **Verify TLS certificate** unchecked (demo mode)

## Collect synthetic attack datasets

```powershell
py scripts\collect_synthetic_data.py --n-rows 20000 --out data\raw\metaverse_synthetic.csv
```

## Evaluate accuracy and false positives

```powershell
py scripts\evaluate_models.py
```

Outputs: `artifacts/evaluation.json`

## Test against simulated DDoS and phishing

```powershell
py scripts\test_attack_scenarios.py
```

## Documentation

- Threat study: `docs/threat_study.md`
- Threat mitigation strategies: `docs/threat_mitigation.md`
