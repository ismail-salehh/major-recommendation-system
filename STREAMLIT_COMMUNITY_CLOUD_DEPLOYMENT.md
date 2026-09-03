# Streamlit Community Cloud Deployment Guide

This guide provides step-by-step instructions for hosting the **University Major Recommendation System** for free on **[Streamlit Community Cloud](https://streamlit.io/cloud)**.

---

## 🏗️ Architecture Overview

- **Web Application Code & UI (`app.py`, `style.css`, `src/`)**: Pushed to your GitHub repository.
- **Machine Learning Models & Encoders**: Hosted on your private Hugging Face repository (`Awsomio/major-recommender`).
- **Streamlit Community Cloud**: Automatically installs dependencies from `requirements.txt`, fetches model weights from Hugging Face Hub using your Hugging Face Token stored in **Streamlit Secrets**, and serves the app.

---

## 📋 Step 1: Push Changes to GitHub

Ensure all updated files (`requirements.txt`, `src/recommender.py`, etc.) are committed and pushed to GitHub:

```bash
git add .
git commit -m "Configure Hugging Face Hub loader and Streamlit Cloud compatibility"
git push origin main
```

---

## 🚀 Step 2: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.
2. Click the **New app** button (or **Deploy an app**).
3. Select **Use existing repo** and fill in your repository details:
   - **Repository**: `YOUR_GITHUB_USERNAME/major-recommendation-system`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: *(Optional)* Customize your app's public URL slug.

---

## 🔑 Step 3: Configure Secrets (Hugging Face Access Token)

Before clicking Deploy, add your Hugging Face Token to **Secrets**:

1. Click **Advanced settings...** at the bottom of the deployment popup (or go to **App settings > Secrets** after creating the app).
2. In the **Secrets** TOML editor, paste the following snippet:

```toml
HUGGINGFACE_TOKEN = "your_huggingface_access_token_here"
```

*(Paste your actual Hugging Face Access Token from your `.env` file into the quotes).*

3. Click **Save**.

---

## ⚡ Step 4: Launch & Monitor

1. Click **Deploy!**
2. Streamlit Cloud will trigger a build:
   - It will install all dependencies listed in `requirements.txt`.
   - It will start `app.py`.
   - `src/recommender.py` will read `st.secrets["HUGGINGFACE_TOKEN"]` and download `major_recommender_ensemble.pkl`, `label_encoders.pkl`, and `model_metadata.json` from Hugging Face Hub.
3. Once loaded, your interactive web application will be live at `https://<your-app-name>.streamlit.app`!

---

## 💡 Performance & Memory Optimization

- **Resource Limit**: Streamlit Community Cloud provides **1 GB to ~3 GB RAM** per app.
- **Resource Caching**: The app utilizes `@st.cache_resource` in `app.py` so that the model ensemble is unpickled into memory **only once** upon server startup. Subsequent user interactions reuse the cached engine instantly without re-downloading or re-loading memory.

---

## 🛠️ Troubleshooting

### 1. `FileNotFoundError` or `Unauthorized (401)` Error
- **Cause**: The `HUGGINGFACE_TOKEN` secret is missing or invalid.
- **Solution**: Go to your app dashboard -> **Settings** (bottom right menu) -> **Secrets**, and verify `HUGGINGFACE_TOKEN = "hf_..."` is correctly saved.

### 2. App Keeps Resetting / "Oh no, your app ran out of memory"
- **Cause**: Loading multiple instances of the model simultaneously.
- **Solution**: `@st.cache_resource` prevents multiple instances. Ensure `load_engine()` in `app.py` retains `@st.cache_resource`.
