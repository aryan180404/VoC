# Deploy in 15 minutes: GitHub + Streamlit Community Cloud

You need: a GitHub account, a Gemini API key (https://aistudio.google.com/apikey).

## Step 1. Put the project on GitHub
1. Unzip `voc-copilot`.
2. On github.com click **New repository**. Name it `voc-copilot`, choose **Public**, do NOT add a README. Create.
3. Upload the files, either way:
   - **Website:** on the new repository page click **uploading an existing file**, drag in everything inside the
     `voc-copilot` folder (including the hidden `.github` and `.streamlit` folders: on Mac press Cmd+Shift+. to show them), then **Commit changes**.
   - **Terminal:**
     ```bash
     cd voc-copilot
     git init && git add . && git commit -m "Voice of Customer Copilot"
     git branch -M main
     git remote add origin https://github.com/<your-username>/voc-copilot.git
     git push -u origin main
     ```
4. Check that **no file contains your key**. `.env` and `.streamlit/secrets.toml` are git-ignored, so they never upload.

## Step 2. Give GitHub the key (to run Gemini on all reviews)
1. Repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.
2. Name: `GEMINI_API_KEY`. Secret: your key. **Add secret**.
3. Repository → **Actions** tab → enable workflows if asked → **Run full pipeline with Gemini** → **Run workflow**.
   - First try with `200` in the "trial" box to check it works (about 2 minutes), then run again with the box empty.
   - The full run takes about 25 to 40 minutes on the free Gemini tier. When it finishes, it commits the Gemini
     results back to the repository.

## Step 3. Put the dashboard online (Streamlit Community Cloud)
1. Go to https://share.streamlit.io and **Continue with GitHub**.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `<your-username>/voc-copilot` · Branch: `main` · Main file path: `app.py`.
4. Optional: App URL, e.g. `voc-copilot-ari`.
5. Open **Advanced settings**: Python version **3.12**. In **Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
6. **Deploy**. First start takes 2 to 4 minutes. Your link: `https://<app-url>.streamlit.app`

Whenever the repository changes (including the Gemini workflow committing results), the app redeploys itself.
To change the key later: app menu (⋮) → **Settings** → **Secrets**.

## Where the key goes, and why both places
| Place | Used for |
|---|---|
| GitHub → Settings → Secrets → Actions | The workflow that labels all 5,000+ reviews with Gemini (too long to run inside a web app) |
| Streamlit Cloud → App settings → Secrets | Live features in the dashboard: "Ask the data" |
| `.env` file on your laptop (optional) | Running everything locally |

Never paste the key into `app.py`, `config.py`, the README, or any file you commit. If a key ever leaks, delete it
in Google AI Studio and create a new one.

## Run on your laptop (optional)
```bash
pip install -r requirements.txt
cp .env.example .env        # paste your key into .env
python run_pipeline.py
streamlit run app.py
```

## Good to know
- Free Streamlit apps sleep after a few days without visitors; the next visitor clicks "Yes, get this app back up".
- Decisions saved in the Decision Engine reset when the cloud app restarts. Use **Export approved** to keep them.
- The answer key in `data/eval/` is synthetic and public in this repository on purpose, so anyone can re-run the
  evaluation. Only `src/evaluate.py` reads it; the labelling code never does.
