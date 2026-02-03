# 🎬 Segment Video App (Databricks + Plotly Dash)

A small Plotly Dash app designed to run as a **Databricks App** and be deployed via **Databricks Asset Bundles (DABs)**.

![segment-video-app-gif](https://github.com/user-attachments/assets/53515de6-9bed-4dd5-af7d-76ee9ddc181e)

It lets a user:

- Upload a video (drag/drop or browse) **or** load an existing video from a Unity Catalog Volume path
- Enter a free-text segmentation prompt (e.g., "weimaraner", "person wearing red shirt")
- Configure processing options:
  - `frame_stride` (default **30**): process every Nth frame (higher = faster, less accurate)
  - `truncate` (default **true**): output only matching segments vs full video
  - `threshold` (default **0.5**): detection sensitivity (0.25-0.9; lower = more sensitive, more false positives)
- Upload the video to a Unity Catalog Volume: `/Volumes/pubsec_video_processing/cv/auto_segment/inputs/`
- Trigger an existing Databricks Job: **auto-segment-video** (Job ID **1063278823055445**) with parameters:
  - `trigger_location`: the full Volume path of the uploaded video
  - `prompt`: the user-entered prompt
- Additional parameters sent to the job:
  - `frame_stride`: integer (stringified)
  - `truncate`: `true`/`false` (stringified)
  - `threshold`: float (stringified)
- Poll the job status and display elapsed time
- When complete, poll for output at `/Volumes/pubsec_video_processing/cv/auto_segment/outputs/` (same filename) and display it in the browser
- Display an **AI-generated description** under the output video (if present), pulled from:
  - `/Volumes/pubsec_video_processing/cv/auto_segment/descriptions/<output_stem>.txt`
  - Example: `outputs/my_video.mp4` → `descriptions/my_video.txt`
- If a user leaves and returns later, they can look up an output video by filename and display the video **and** description.
- Serve output files through a lightweight `/download` route (streamed from the UC volume via `w.files.download`) to avoid embedding large blobs in the page.

## 📁 Project Structure

```
segment-video-app/
├── app/
│   ├── app.py              # Dash application
│   ├── requirements.txt    # Python dependencies
│   ├── app.yml             # Databricks App configuration
│   └── assets/
│       └── custom.css      # Custom styling
├── notebooks/
│   ├── job-auto-segment-video.ipynb       # Main job notebook
│   ├── job-auto-segment-video-fmapi.ipynb # Alternative job using FM API
│   ├── model-sam3-video.ipynb             # SAM 3 video model experiments
│   ├── model-sam3.ipynb                   # SAM 3 model experiments
│   ├── model-blip.ipynb                   # BLIP model experiments
│   ├── model-gemini3flash.ipynb           # Gemini 3 Flash experiments
│   ├── model-qwen3.ipynb                  # Qwen 3 experiments
│   └── workspace-setup.ipynb              # Initial workspace setup
├── env.example             # Example env vars for local dev
├── databricks.yml          # Databricks Asset Bundle (DAB) definition
└── README.md               # This file
```

## 🚀 Local Development

1) Create a venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

2) Configure environment variables:

```bash
cp env.example .env
export $(cat .env | xargs)
```

3) Run the app:

```bash
python app/app.py
```

Then open `http://localhost:8050`.

## 🚀 Deploying to Databricks

To deploy this app and the associated job to Databricks using Asset Bundles:

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to dev environment (default)
databricks bundle deploy

# Deploy to prod environment
databricks bundle deploy --target prod
```

The `databricks.yml` file defines:
- The Dash app deployment configuration
- The auto-segment-video job with file arrival trigger
- Environment-specific settings (dev/prod)

## 📓 Notebooks

The `notebooks/` folder contains:

- **job-auto-segment-video.ipynb**: Main production job for video segmentation
- **job-auto-segment-video-fmapi.ipynb**: Alternative implementation using Foundation Model API
- **model-sam3-video.ipynb**: Experiments with SAM 3 for video segmentation
- **model-sam3.ipynb**: SAM 3 model testing and evaluation
- **model-blip.ipynb**: BLIP model for image captioning and description generation
- **model-gemini3flash.ipynb**: Gemini 3 Flash model experiments
- **model-qwen3.ipynb**: Qwen 3 model experiments
- **workspace-setup.ipynb**: Initial setup and configuration for the Databricks workspace

## 🔐 Auth Notes

This app uses the `databricks-sdk` and expects authentication via environment variables (for local), or the Databricks Apps runtime (for deployed).

Helpful references:

- Databricks Asset Bundles: `https://docs.databricks.com/aws/en/dev-tools/bundles?utm_source=openai`
- Databricks SDK for Python (Files in UC volumes): `https://docs.databricks.com/aws/en/dev-tools/sdk-python#manage-files-in-unity-catalog-volumes`
