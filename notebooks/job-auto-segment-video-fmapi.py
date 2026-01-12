# Databricks notebook source
trigger_location = dbutils.widgets.get("trigger_location")
prompt = dbutils.widgets.get("prompt")
frame_stride = int(dbutils.widgets.get("frame_stride"))
truncate = dbutils.widgets.get("truncate")

# trigger_location = '/Volumes/pubsec_video_processing/cv/auto_segment/inputs/usps_youtube_example.mov'
# prompt = 'employees or postal workers handling or delivering mail'
# frame_stride = 30
# truncate = True

if truncate==0 or truncate=='false' or truncate=='False':
  truncate = False
else:
  truncate=True

print(trigger_location)
print(prompt)
print(frame_stride)
print(truncate)
print(type(truncate))

# COMMAND ----------

# dbutils.notebook.exit("Notebook execution stopped by user request.")

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

# COMMAND ----------

from datetime import datetime

if trigger_location.endswith('/') and (trigger_location[-4]!='.' or trigger_location[-5]!='.'):
  # Your volume directory path
  directory_path = trigger_location

  # List all files in the directory
  files = dbutils.fs.ls(directory_path)

  # Filter out directories, keep only files
  files = [f for f in files if not f.isDir()]

  # Sort by modification time (most recent first)
  files_sorted = sorted(files, key=lambda x: x.modificationTime, reverse=True)

  # Get the most recent file
  if files_sorted:
      most_recent_file = files_sorted[0]
      most_recent_path = most_recent_file.path.replace('dbfs:','')
      most_recent_name = most_recent_file.name
      
      print(f"Most recent file: {most_recent_name}")
      print(f"Full path: {most_recent_path}")
      print(f"Modified: {datetime.fromtimestamp(most_recent_file.modificationTime/1000)}")
  else:
      print("No files found in directory")
else:
  most_recent_file = dbutils.fs.ls(trigger_location)[0]
  most_recent_path = most_recent_file.path.replace('dbfs:','')
  most_recent_name = most_recent_file.name

  print(f"Most recent file: {most_recent_name}")
  print(f"Full path: {most_recent_path}")
  print(f"Modified: {datetime.fromtimestamp(most_recent_file.modificationTime/1000)}")

# COMMAND ----------

# most_recent_file = trigger_location
# most_recent_name = most_recent_file.split("/")[-1]
# most_recent_path = most_recent_file.replace('dbfs:','')

# COMMAND ----------

from huggingface_hub import login
import os

hf_pat = dbutils.secrets.get("justinm-buildathon-secrets", "hf_pat")
os.environ["HF_TOKEN"] = hf_pat
login(token=hf_pat)

# COMMAND ----------

import mlflow

model = mlflow.pyfunc.load_model("models:/pubsec_video_processing.cv.transformers-sam3-video@job")

# COMMAND ----------

def write_results(FILE_URL, results):
  import os

  if FILE_URL.startswith("/Volumes/pubsec_video_processing/cv/auto_segment/inputs/"):
    OUTPUT_FILE_URL = FILE_URL.replace("inputs", "outputs")
    output_dir = os.path.dirname(OUTPUT_FILE_URL)
  else:
    OUTPUT_FILE_URL = most_recent_path.replace(most_recent_name, f"outputs/{most_recent_name}")
    output_dir = os.path.dirname(OUTPUT_FILE_URL)
  os.makedirs(output_dir, exist_ok=True)
  FPS = 24

  def decode_mask(encoded_mask: str) -> np.ndarray:
      """Decode base64 mask back to numpy array"""
      buf = BytesIO(base64.b64decode(encoded_mask))
      return np.load(buf)

  def add_timestamp(frame: np.ndarray, timestamp_sec: float) -> np.ndarray:
      """Add timestamp overlay to frame in the top-right corner"""
      # Convert seconds to HH:MM:SS.ms format
      hours = int(timestamp_sec // 3600)
      minutes = int((timestamp_sec % 3600) // 60)
      seconds = int(timestamp_sec % 60)
      milliseconds = int((timestamp_sec % 1) * 1000)
      
      timestamp_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
      
      # Create a copy to avoid modifying original
      frame_with_timestamp = frame.copy()
      
      # Get frame dimensions
      height, width = frame.shape[:2]
      
      # Set up text properties
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.7
      font_thickness = 2
      text_color = (255, 255, 255)  # White text
      bg_color = (0, 0, 0)  # Black background
      padding = 10
      
      # Get text size
      (text_width, text_height), baseline = cv2.getTextSize(
          timestamp_text, font, font_scale, font_thickness
      )
      
      # Position in top-right corner
      text_x = width - text_width - padding
      text_y = padding + text_height
      
      # Draw semi-transparent background rectangle
      bg_x1 = text_x - 5
      bg_y1 = text_y - text_height - 5
      bg_x2 = text_x + text_width + 5
      bg_y2 = text_y + baseline + 5
      
      # Create overlay for semi-transparency
      overlay = frame_with_timestamp.copy()
      cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
      cv2.addWeighted(overlay, 0.6, frame_with_timestamp, 0.4, 0, frame_with_timestamp)
      
      # Draw text
      cv2.putText(
          frame_with_timestamp,
          timestamp_text,
          (text_x, text_y),
          font,
          font_scale,
          text_color,
          font_thickness,
          cv2.LINE_AA
      )
      
      return frame_with_timestamp

  # Open original video to get frames
  print("Processing frames and applying masks...")
  cap = cv2.VideoCapture(FILE_URL)
  fps = cap.get(cv2.CAP_PROP_FPS) or FPS

  # Create a mapping of frame_idx to results
  print(f"Found {len(results)} frames in results")
  result_map = {r["frame_idx"]: r for r in results}

  frame_idx = 0
  saved_images = []
  segmented_images = []

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      
      # Only process frames with segmentation results
      if frame_idx in result_map:
          # Convert BGR to RGB
          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          
          res = result_map[frame_idx]
          
          # if res["masks"]:
          #     # Get the first (highest score) mask
          #     mask = decode_mask(res["masks"][0])
              
          #     # Overlay with transparency
          #     overlay = rgb_frame.copy()
          #     overlay[mask > 0.5] = [0, 255, 0]  # Green overlay
          #     masked_frame = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)

          if res["masks"]:
              # Get the first (highest score) mask
              masks = [decode_mask(x) for x in res["masks"]]
              
              # Overlay with transparency
              overlay = rgb_frame.copy()
              for mask in masks:
                  overlay[mask > 0.5] = [0, 255, 0]  # Green overlay
              masked_frame = cv2.addWeighted(rgb_frame, 0.7, overlay, 0.3, 0)
              
          else:
              masked_frame = rgb_frame
          
          # Calculate timestamp for this frame
          timestamp_sec = frame_idx / fps
          
          # Add timestamp overlay (convert back to BGR for cv2 operations, then back to RGB)
          masked_frame_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
          masked_frame_with_timestamp = add_timestamp(masked_frame_bgr, timestamp_sec)
          masked_frame = cv2.cvtColor(masked_frame_with_timestamp, cv2.COLOR_BGR2RGB)
          
          # Save frame
          if not truncate:
              saved_images.append(Image.fromarray(rgb_frame))
              segmented_images.append(Image.fromarray(masked_frame))
          elif res["masks"]:
              saved_images.append(Image.fromarray(rgb_frame))
              segmented_images.append(Image.fromarray(masked_frame))
      
      frame_idx += 1

  cap.release()
  print(f"Saved {len(saved_images)} frames to memory")
  print(f"Saved {len(segmented_images)} segmented frames to memory")

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
      segmented_images,
      fps=(fps/frame_stride) * 5,  # Use original FPS instead of hardcoded 24
      codec='libx264',
      pixelformat='yuv420p'
  )

  temp_size = os.path.getsize(temp_video_path)
  print(f"Temporary video created: {temp_size:,} bytes ({temp_size/1024/1024:.2f} MB)")

  # Copy to Volumes
  print(f"Copying to Volumes: {OUTPUT_FILE_URL}")
  shutil.copy2(temp_video_path, OUTPUT_FILE_URL)

  final_size = os.path.getsize(OUTPUT_FILE_URL)
  print(f"✓ Video successfully saved to: {OUTPUT_FILE_URL}")
  print(f"  Final size: {final_size:,} bytes ({final_size/1024/1024:.2f} MB)")

  # Clean up temporary file
  if os.path.exists(temp_video_path):
      os.remove(temp_video_path)
      print("Cleaned up temporary file")

  return OUTPUT_FILE_URL

# COMMAND ----------

def process_file(triggered_file, prompt):
    print(f"Triggered by file: {triggered_file}")
    # Your processing logic here
    model_input = pd.DataFrame([{
        "video_path": triggered_file,
        "prompt": prompt,
        "frame_stride": frame_stride,  # Process every nth frame
        "batch_size": 4,
        "threshold": 0.5,
        "mask_threshold": 0.5
    }])
    results = model.predict(model_input)
    return results

# COMMAND ----------

print("Files processed in this run:", most_recent_path)

print("Segmenting video...")
starting_time = timeit.default_timer()
results = process_file(most_recent_path, prompt)
print(f"Inference time: {round((timeit.default_timer() - starting_time))} secs")

# COMMAND ----------

output_file_url = write_results(most_recent_path, results)

# COMMAND ----------

import io
import numpy
import base64
import cv2
import asyncio
import numpy as np
from openai import AsyncOpenAI, ChatCompletion
from typing import NamedTuple, List, Optional, Generator, Tuple

# helpers
class FrameBatch(NamedTuple):
    content: List[dict]
    sizes: List[int]
    total_bytes: int
    
    @property
    def frame_count(self) -> int:
        return len(self.content)

def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """cv2.imencode is 3-5x faster than PIL, no color conversion needed."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def to_content(jpg_bytes: bytes) -> dict:
    b64 = base64.b64encode(jpg_bytes).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}

def fit_to_size(frame: np.ndarray, max_size: int) -> Tuple[bytes, int]:
    
    # Early exit - most frames fit at q=95
    data = encode_jpeg(frame, 95)
    if len(data) <= max_size:
        return data, 95
    
    # binary search FTW!
    lo, hi, best = 10, 94, (encode_jpeg(frame, 10), 10)
    while lo <= hi:
        mid = (lo + hi) >> 1
        data = encode_jpeg(frame, mid)
        if len(data) <= max_size:
            best, lo = (data, mid), mid + 1
        else:
            hi = mid - 1
    return best

def process_frame(
    frame: np.ndarray, 
    quality: Optional[int] = None,
    max_size: int = 500_000
) -> Tuple[dict, int, int]:
    if quality:
        data = encode_jpeg(frame, quality)
    else:
        data, quality = fit_to_size(frame, max_size)
    return to_content(data), quality, len(data)

def stream_content(
    video: cv2.VideoCapture,
    quality: Optional[int] = None,
    max_size: int = 500_000
) -> Generator[Tuple[dict, int, int], None, None]:
    while True:
        ok, frame = video.read()
        if not ok:
            return
        yield process_frame(frame, quality, max_size)

def batch_content(
    video: cv2.VideoCapture,
    quality: Optional[int] = None,
    max_frame_size: int = 500_000,
    max_batch_size: int = 3_000_000
) -> Generator[FrameBatch, None, None]:
    max_frame_size = min(max_frame_size, max_batch_size)
    batch, sizes, batch_size = [], [], 0
    for content, _, size in stream_content(video, quality, max_frame_size):
        if batch and batch_size + size > max_batch_size:
            yield FrameBatch(batch, sizes, batch_size)
            batch, sizes, batch_size = [], [], 0
        batch.append(content)
        sizes.append(size)
        batch_size += size
    
    if batch:
        yield FrameBatch(batch, sizes, batch_size)

def summarize_frames(frame_batch: FrameBatch):
    return oai.chat.completions.create(
    model=FMAPI_MODEL,
    messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 
                'Describe what is happening in the following sequence of images '
                'which represent a frames from a video. Describe what is going on in from the cameras '
                'perspective. Be descriptive - but succinct and functional'
            },
            *frame_batch.content
        ]}
    ]
)
    
def get_content(completion: ChatCompletion) -> str:
    '''
        get_content extracts the textual content.  Required with gemini 3 models
        as g3 model's content may (or may not) include a thoughtSignature which 
        is a reference to the model's reasoning process.
    '''
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0]['text']
    else:
        print(type(content))
        print(content)
        raise Exception('weird content signature')

# COMMAND ----------

# actual running code
vid_path = output_file_url
video = cv2.VideoCapture(vid_path)
frame_batches = list(batch_content(video, quality=80))
video.release()

FMAPI_SERVING_URL = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'
FMAPI_BASE_URL = f'{FMAPI_SERVING_URL}/serving-endpoints'
print(f'🎯 FMAPI_BASE_URL: {FMAPI_BASE_URL}')
FMAPI_API_TOKEN = dbutils.secrets.get('justinm-buildathon-secrets', 'db_pat')
# Serving endpoint I created using my own GCP acc't
# to side-step issues w/ limits databricks hosted model issues.
FMAPI_MODEL = 'stsu-gemini-3-flash' 

# clients
oai = AsyncOpenAI(
    api_key = FMAPI_API_TOKEN,
    base_url = FMAPI_BASE_URL
)

# Get results
ai_results = await asyncio.gather(*[
    summarize_frames(frame_batch)
    for frame_batch in frame_batches
])

# concatenation of previous generations
context = []
for idx, result in enumerate(ai_results):
    description = f'SECTION {idx}\n' + get_content(result)
    context.append(description)

context = '\n'.join(context)

video_summary = await oai.chat.completions.create(
    model=FMAPI_MODEL,
    messages=[
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 
                'Summarize the following sections of a video with a focus on '
                'what is functionally happening over time. Be descriptive. '
                'The resulting summary should be a functional narrative. '
                'Return this as Markdown text - and only markdown. '
            },
            {'type': 'text', 'text': context}
        ]}
    ]
)

text = get_content(video_summary)
print(text)

# COMMAND ----------

txt_filename = most_recent_path.split('.')[0] + '.txt'
txt_filename = txt_filename.replace('/inputs/', '/descriptions/')
print(txt_filename)

dbutils.fs.put(txt_filename, text, True)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# captioning_model = mlflow.pyfunc.load_model("models:/pubsec_video_processing.cv.transformers-blip@job")

# COMMAND ----------

# from PIL import Image
# from io import BytesIO
# import base64

# def pil_to_base64_str(img: Image.Image, format: str = "PNG") -> str:
#     """
#     Convert a PIL image to a Base64-encoded string.
    
#     Args:
#         img: PIL.Image.Image
#         format: image format, e.g., 'PNG' or 'JPEG'
        
#     Returns:
#         str: Base64 string that can be safely passed in JSON
#     """
#     buf = BytesIO()
#     img.save(buf, format=format)
#     buf.seek(0)
#     b64_str = base64.b64encode(buf.read()).decode("utf-8")
#     return b64_str

# COMMAND ----------

# from io import BytesIO

# written_images_str = [pil_to_base64_str(x) for x in written_images]

# COMMAND ----------

# model_input = {
#     "image_path": written_images_str
# }

# captions = captioning_model.predict(model_input)

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# # Initialize the client
# w = WorkspaceClient()

# # Define your endpoint name
# endpoint_name = "databricks-gpt-oss-120b"

# # # Create a chat message
# # messages = [
# #     ChatMessage(
# #         role=ChatMessageRole.USER,
# #         content=f"The following is a list of descriptions of images. The images represent frames from a video, in order. Can you identify anything out of the ordinary or anomalous in the video? When I say anomlous, I mean actions taking place that are against the law or violent in nature. If you find something that shouldn't be happening, describe why it is anomolous. If you don't find anything abnormal, just respond with 'Nothing anomolous found.' \n\n {captions}"
# #     )
# # ]
# # Create a chat message
# messages = [
#     ChatMessage(
#         role=ChatMessageRole.USER,
#         content=f"The following is a list of descriptions of images. The images represent frames from a video, in order. There may be obvious scene changes based on the descriptions. Please summarize the contents of each scene in the video in a single sentence each.' \n\n {captions}"
#     )
# ]

# # Query the endpoint
# response = w.serving_endpoints.query(
#     name=endpoint_name,
#     messages=messages,
#     # max_tokens=500  # optional parameter
# )

# # Access the response
# r = response.choices[0].message.content
# # print(r)
# text = [x for x in r if x['type'] == 'text'][0]['text']
# print(text)

# COMMAND ----------

# txt_filename = most_recent_path.split('.')[0] + '.txt'
# txt_filename = txt_filename.replace('/inputs/', '/descriptions/')
# print(txt_filename)

# dbutils.fs.put(txt_filename, text, True)

# COMMAND ----------

