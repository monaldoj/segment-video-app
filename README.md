# 🎬 Segment Video App (Databricks + Plotly Dash)

A small Plotly Dash app designed to run as a **Databricks App** and be deployed via **Databricks Asset Bundles (DABs)**.

It lets a user:

- Upload a video (drag/drop or browse)
- Enter a free-text segmentation prompt
- Configure processing options:
  - `frame_stride` (default **5**): process every Nth frame
  - `truncate` (default **false**): output full video vs only matching segments
- Upload the video to a Unity Catalog Volume: `/Volumes/justinm/cv/auto_segment/inputs/`
- Trigger an existing Databricks Job: **auto-segment-video** (Job ID **774232501138119**) with parameters:
  - `trigger_location`: the full Volume path of the uploaded video
  - `prompt`: the user-entered prompt
- Additional parameters sent to the job:
  - `frame_stride`: integer (stringified)
  - `truncate`: `true`/`false` (stringified)
- Poll the job status and display elapsed time
- When complete, poll for output at `/Volumes/justinm/cv/auto_segment/outputs/` (same filename) and display it in the browser
- Display an **AI-generated description** under the output video (if present), pulled from:
  - `/Volumes/justinm/cv/auto_segment/descriptions/<output_stem>.txt`
  - Example: `outputs/my_video.mp4` → `descriptions/my_video.txt`
- If a user leaves and returns later, they can look up an output video by filename and display the video **and** description.
- Serve output files through a lightweight `/download` route (streamed from the UC volume via `w.files.download`) to avoid embedding large blobs in the page.

## 📁 Project Structure

```
segment-video-app/
├── app/
│   ├── app.py              # Dash application
│   ├── requirements.txt    # Python dependencies
│   └── app.yml             # Databricks App configuration
├── env.example             # Example env vars for local dev
├── databricks.yml          # Databricks Asset Bundle (DAB) definition
├── deploy.sh               # Convenience deploy script (DAB)
└── run_local.sh            # Convenience local run script
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

## 🔐 Auth Notes

This app uses the `databricks-sdk` and expects authentication via environment variables (for local), or the Databricks Apps runtime (for deployed).

Helpful references:

- Databricks Asset Bundles: `https://docs.databricks.com/aws/en/dev-tools/bundles?utm_source=openai`
- Databricks SDK for Python (Files in UC volumes): `https://docs.databricks.com/aws/en/dev-tools/sdk-python#manage-files-in-unity-catalog-volumes`


