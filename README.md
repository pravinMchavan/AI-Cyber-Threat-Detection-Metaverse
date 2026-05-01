# AI-Cyber-Threat-Detection-Metaverse (MCA Final Year Project)

This is a final year MCA project.

MVP (first working version):
- Detect suspicious/anomalous network activity from a CSV file
- Show prediction results in a small demo app (Streamlit)

Simple meaning of words:
- MVP = smallest working version you can demo
- CSV = table file (rows/columns) like Excel
- Model = trained program that predicts Normal vs Attack

## Folder structure

```
AI-Cyber-Threat-Detection-Metaverse/
	artifacts/            # outputs (trained model, metrics) [auto-created]
	data/
		raw/                # put your raw CSV datasets here
		processed/          # optional: cleaned/processed CSVs
	scripts/
		train_model.py      # trains the baseline model
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
