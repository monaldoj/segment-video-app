# Databricks notebook source
# %pip install opencv-python

# COMMAND ----------

# MAGIC %pip install imageio imageio[ffmpeg] imageio[pyav]
# MAGIC %restart_python

# COMMAND ----------

from huggingface_hub import login
import os

hf_pat = dbutils.secrets.get("justin-fe-secrets", "hf_pat")
os.environ["HF_TOKEN"] = hf_pat
login(token=hf_pat)

# COMMAND ----------

import os
import mlflow.pyfunc
import torch
import numpy as np
import pandas as pd
import cv2
import base64
from io import BytesIO
from PIL import Image
from transformers import Sam3Processor, Sam3Model


class SAM3Video(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for SAM3 image + video segmentation with batching
    """

    # -------------------------
    # Model loading
    # -------------------------
    def load_context(self, context):
        from huggingface_hub import login
        from transformers import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()

        hf_pat = os.environ["HF_TOKEN"]
        login(token=hf_pat)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=dtype
        ).to(self.device)

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    # -------------------------
    # Utils
    # -------------------------
    def _encode_mask(self, mask: np.ndarray) -> str:
        """Encode float mask → base64"""
        buf = BytesIO()
        np.save(buf, mask.astype(np.float32))
        return base64.b64encode(buf.getvalue()).decode()

    def _video_capture(self, path):
        if path.startswith("http"):
            return cv2.VideoCapture(path)
        return cv2.VideoCapture(os.path.expanduser(path))

    # -------------------------
    # Core video processing
    # -------------------------
    def _process_video(
        self,
        video_path: str,
        prompt: str,
        frame_stride: int,
        batch_size: int,
        threshold: float,
        mask_threshold: float
    ):
        cap = self._video_capture(video_path)

        frames = []
        frame_indices = []
        results = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                frame_indices.append(idx)

            if len(frames) == batch_size:
                results.extend(
                    self._run_batch(
                        frames,
                        frame_indices,
                        prompt,
                        threshold,
                        mask_threshold
                    )
                )
                frames, frame_indices = [], []

            idx += 1

        # leftover frames
        if frames:
            results.extend(
                self._run_batch(
                    frames,
                    frame_indices,
                    prompt,
                    threshold,
                    mask_threshold
                )
            )

        cap.release()
        return results

    # -------------------------
    # Batched SAM3 inference
    # -------------------------
    def _run_batch(
        self,
        images,
        frame_indices,
        prompt,
        threshold,
        mask_threshold
    ):
        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt"
        ).to(self.device)

        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].to(self.model.dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist()
        )

        batch_results = []
        for i, res in enumerate(processed):
            batch_results.append({
                "frame_idx": frame_indices[i],
                "scores": res["scores"].cpu().tolist(),
                "masks": [
                    self._encode_mask(m.cpu().numpy())
                    for m in res["masks"]
                ]
            })

        return batch_results

    # -------------------------
    # MLflow predict
    # -------------------------
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            row = model_input.iloc[0].to_dict()
        else:
            row = model_input

        video_path = row["video_path"]
        prompt = row["prompt"]

        frame_stride = int(row.get("frame_stride", 1))
        batch_size = int(row.get("batch_size", 4))
        threshold = float(row.get("threshold", 0.5))
        mask_threshold = float(row.get("mask_threshold", 0.5))

        return self._process_video(
            video_path=video_path,
            prompt=prompt,
            frame_stride=frame_stride,
            batch_size=batch_size,
            threshold=threshold,
            mask_threshold=mask_threshold
        )


# COMMAND ----------

# Load model and get predictions
print("Loading MLflow model...")
# model = mlflow.pyfunc.load_model(MODEL_URI)
model = SAM3Video()
model.load_context(context=None)

# COMMAND ----------

import mlflow
import imageio
import numpy as np
import pandas as pd
import cv2
import base64
import timeit
from io import BytesIO
from PIL import Image
import os

# Configuration
MODEL_URI = "your_model_uri_here"  # e.g., "models:/sam3_video/1" or "runs:/run_id/model"
video_name = "maren_jack"

VIDEO_PATH = f"/Volumes/pubsec_video/cv/images/{video_name}.MOV"  # Your input video
PROMPT = "boy in white sweater with black stripes"  # Your segmentation prompt
# OUTPUT_FRAMES_DIR = "/Volumes/pubsec_video/cv/images/bruno1_output_dir/"
OUTPUT_VIDEO_PATH = f"/Volumes/pubsec_video/cv/images/{video_name}_output.mp4"
FPS = 30  # Adjust to match your video's FPS

# Create output directory
# os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# Prepare model input
model_input = pd.DataFrame([{
    "video_path": VIDEO_PATH,
    "prompt": PROMPT,
    "frame_stride": 5,  # Process every nth frame
    "batch_size": 4,
    "threshold": 0.5,
    "mask_threshold": 0.5
}])

print("Running inference...")
starting_time = timeit.default_timer()
results = model.predict(context=None, model_input=model_input)
print(f"Inference time: {round((timeit.default_timer() - starting_time))} secs")

# COMMAND ----------

type(results)
result_map = {r["frame_idx"]: r for r in results}
print(result_map.keys())
print(result_map[5])

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# specify the location the model will be saved/registered in Unity Catalog
catalog = "pubsec_video"
schema = "cv"
model_name = "transformers-sam3-video"
model_full_name = f"{catalog}.{schema}.{model_name}"
mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(model_input=model_input, model_output=results)

# Define conda environment with dependencies
conda_env = {
    'channels': ['conda-forge', 'defaults'],
    'dependencies': [
        'python=3.12.3',
        'pip',
        {
            'pip': [
                'mlflow>=2.10.0',
                'torch>=2.0.0',
                'git+https://github.com/huggingface/transformers.git',
                'Pillow',
                'torchvision',
                "cloudpickle==3.0.0",
                # 'pillow>=9.0.0',
                'numpy>=1.23.0',
                'pandas>=1.5.0',
                'accelerate>=0.20.0'
            ]
        }
    ],
    'name': 'sam3_tracker_env'
}

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SAM3Video(),
        signature=signature,
        conda_env=conda_env,
        # extra_pip_requirements=[
        #   "torch",
        #   "git+https://github.com/huggingface/transformers.git",
        #   "Pillow"
        # ]
    )
    
    run_id = run.info.run_id
    print(f"Model registered! URI: runs:/{run_id}/model")

# COMMAND ----------

# register the model using the "run" from above.
mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_full_name)

# COMMAND ----------

# 2. Preview sample frames
import matplotlib.pyplot as plt

def decode_mask(encoded_mask: str) -> np.ndarray:
    """Decode base64 mask back to numpy array"""
    buf = BytesIO(base64.b64decode(encoded_mask))
    return np.load(buf)

def overlay_masks_on_frame(frame, masks, scores, alpha=0.5, score_threshold=0.5):
    """Overlay segmentation masks on a frame with different colors"""
    overlay = frame.copy()
    
    # Filter masks by score
    valid_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
    
    # Generate colors for each mask
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_indices)))[:, :3] * 255
    
    for idx, mask_idx in enumerate(valid_indices):
        mask = masks[mask_idx]
        color = colors[idx].astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0.5] = color
        
        # Blend with frame
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Optional: Add contours
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
    
    return overlay
def display_sample_frames(
    original_video_path: str,
    prediction_output: list,
    num_samples: int = 5,
    alpha: float = 0.5
):
    """Display sample frames with segmentation overlays"""
    
    cap = cv2.VideoCapture(original_video_path)
    prediction_map = {pred["frame_idx"]: pred for pred in prediction_output}
    
    # Select evenly spaced frames that have predictions
    available_frames = sorted(prediction_map.keys())
    if len(available_frames) == 0:
        print("No predictions found!")
        return
    
    step = max(1, len(available_frames) // num_samples)
    sample_indices = available_frames[::step][:num_samples]
    
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(20, 4))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for ax, frame_idx in zip(axes, sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            pred = prediction_map[frame_idx]
            masks = [decode_mask(m) for m in pred["masks"]]
            scores = pred["scores"]
            
            overlay = overlay_masks_on_frame(frame, masks, scores, alpha)

            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            ax.imshow(overlay_rgb)
            ax.set_title(f"Frame {frame_idx}\n{len(scores)} objects")
            ax.axis('off')
    
    cap.release()
    plt.tight_layout()
    plt.show()

display_sample_frames(
    original_video_path=video_path,
    prediction_output=results,
    num_samples=5,
    alpha=0.6
)

# COMMAND ----------

print(len(results))

# COMMAND ----------

# OUTPUT_FRAMES_DIR = "/Volumes/pubsec_video/cv/images/bruno1_output_dir/"
OUTPUT_VIDEO_PATH = f"/Volumes/pubsec_video/cv/images/{video_name}_output2.mp4"

# Open original video to get frames
print("Processing frames and applying masks...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or FPS

# Create a mapping of frame_idx to results
result_map = {r["frame_idx"]: r for r in results}

frame_idx = 0
saved_frames = []
saved_images = []

i = 0 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # If this frame has segmentation results, apply the mask
    if frame_idx in result_map:
        # i+=1
        # print('Processing frame:', i, 'of', len(result_map))
        res = result_map[frame_idx]
        
        if res["masks"]:
            # Get the first (highest score) mask
            mask = decode_mask(res["masks"][0])
            
            # Create visualization: overlay mask on original frame
            # Option 1: Show only segmented object
            # masked_frame = rgb_frame * mask[..., None]
            
            # Option 2: Overlay with transparency
            overlay = rgb_frame.copy()
            overlay[mask > 0.5] = [0, 255, 0]  # Green overlay
            masked_frame = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)
            
            # Option 3: Show mask as binary
            # masked_frame = (mask[..., None] * 255).astype(np.uint8).repeat(3, axis=2)
        else:
            masked_frame = rgb_frame
    # else:
    #     masked_frame = rgb_frame
    
    # Save frame
    saved_images.append(Image.fromarray(masked_frame))
    # frame_path = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{frame_idx:05d}.png")
    # Image.fromarray(masked_frame).save(frame_path)
    # saved_frames.append(frame_path)
    
    frame_idx += 1

cap.release()
print(f"Saved {len(saved_images)} frames to memory") #{OUTPUT_FRAMES_DIR}")

# 3. Create full segmented video
import imageio
import os
import shutil
import tempfile

print("Writing video to temporary file...")
with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
    temp_video_path = tmp_file.name

imageio.mimsave(
    temp_video_path,
    saved_images,
    fps=24,
    codec='libx264',
    pixelformat='yuv420p'
)

temp_size = os.path.getsize(temp_video_path)
print(f"Temporary video created: {temp_size:,} bytes ({temp_size/1024/1024:.2f} MB)")

# Copy to Volumes
print(f"Copying to Volumes: {OUTPUT_VIDEO_PATH}")
shutil.copy2(temp_video_path, OUTPUT_VIDEO_PATH)

final_size = os.path.getsize(OUTPUT_VIDEO_PATH)
print(f"✓ Video successfully saved to: {OUTPUT_VIDEO_PATH}")
print(f"  Final size: {final_size:,} bytes ({final_size/1024/1024:.2f} MB)")

# Clean up temporary file
if os.path.exists(temp_video_path):
    os.remove(temp_video_path)
    print("Cleaned up temporary file")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

