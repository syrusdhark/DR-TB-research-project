# Streamlit Community Cloud Deployment

Use this checklist to publish the DR-TB Prediction System from any machine.

## 1. Prerequisites

1. GitHub repo: `syrusdhark/DR-TB-research-project` (main branch, `app.py` at repo root).
2. Model checkpoint: `results/models/exact_match_nov2025.pth` (≈76 MB, below GitHub’s 100 MB limit).
3. Python dependencies pinned inside `requirements.txt`.

## 2. Create the App on Streamlit Cloud

1. Visit <https://share.streamlit.io>, sign in with GitHub, and authorize access to the repo.
2. Click **“New app”** and select:
   - **Repository:** `syrusdhark/DR-TB-research-project`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Leave the default Python version (3.12) or pin it under **Advanced settings** if needed.
4. Press **Deploy**. The build installs `requirements.txt`, downloads the 76 MB checkpoint, and launches Streamlit.

## 3. Secrets & Environment Variables

The current code auto-detects the latest checkpoint inside `results/models`. No secrets are required, but you can configure optional values under **App → Settings → Secrets**:

```toml
[app]
MODEL_PATH = "results/models/exact_match_nov2025.pth"
THRESHOLD = 0.638
```

You can also toggle telemetry suppression by setting `STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"` in **Advanced settings → Environment variables**.

## 4. Build Tips

- Streamlit runners have limited RAM/GPU. The model loads on CPU automatically if CUDA is unavailable.
- Build logs show PyTorch wheel downloads; expect ~3–4 minutes for the first deploy.
- If the build fails because of missing system libraries, add them to `packages.txt` (e.g., `libgl1`, `libgomp1`) and commit.

## 5. Smoke Test After Deployment

1. Open the deployed URL (e.g., `https://<your-handle>-dr-tb-research-project.streamlit.app`).
2. Upload a sample chest X-ray (PNG/JPG <10 MB).
3. Fill a small clinical profile and run **“🔬 Run Prediction”**.
4. Toggle **“Show Detailed Report”** and download the TXT report to confirm full functionality.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Unable to deploy. The app's main file app.py has not been pushed` | Ensure latest commit pushed to GitHub (done as of `2e279fe`). |
| Build keeps restarting | Reduce concurrent users or upgrade to Streamlit paid tier. |
| `CUDA out of memory` | Streamlit Cloud uses CPU; ignore or set `TORCH_ENABLE_MPS_FALLBACK=1`. |
| Model missing | Confirm `results/models/exact_match_nov2025.pth` exists in repo or download from external storage at startup. |

## 7. Update Workflow

1. Make local changes.
2. `git commit` + `git push main`.
3. Streamlit Cloud auto-rebuilds; monitor **App → Logs**.
4. Record notable updates in `IMPLEMENTATION_STATUS.md` or release notes.

This process keeps the deployment reproducible and allows you to redeploy from any machine without manually copying files.

