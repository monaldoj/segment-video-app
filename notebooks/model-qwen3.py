# Databricks notebook source
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import pandas as pd
from typing import Union, List
import io
import base64
from io import BytesIO
import requests

class Qwen3CaptioningModel(PythonModel):
    """
    MLflow PyFunc wrapper for Qwen3-VL-2B-Instruct image captioning model.
    
    This model accepts images in multiple formats:
    - PIL Image objects
    - File paths (strings)
    - Base64 encoded strings
    - Bytes
    - URLs
    """
    
    def load_context(self, context):
        """
        Load the Qwen3-VL model and processor.
        
        Args:
            context: MLflow context containing artifacts and parameters
        """
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        
        # Load the model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
    
    def _convert_image(self, image_path) -> str:
        """
        Convert various image input formats to a format compatible with Qwen3-VL.
        
        Args:
            image_path: Image in various formats (path, bytes, base64, PIL Image, URL)
            
        Returns:
            String representation suitable for Qwen3-VL (file path or URL)
        """
        # print("IMAGE PATH:", image_path)
        if isinstance(image_path, Image.Image):
            # print("IT IS AN IMAGE")
            # Convert PIL Image to base64 data URL
            buffered = BytesIO()
            image_path.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        elif isinstance(image_path, str):
            # print("FIRST LEVEL STRING")
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # HTTP/HTTPS URL - return as is
                # print("IT IS AN HTTP")
                return image_path
            elif image_path.startswith('/') or image_path.startswith('file://'):
                # Local file path
                # print("IT IS A /")
                if not image_path.startswith('file://'):
                    image_path = f"file://{image_path}"
                return image_path
            elif image_path.startswith('data:image'):
                # Already a base64 data URL
                # print("IT IS A base64str")
                return image_path
            else:
                # Assume it's a base64 string without header
                # print("IT IS A str to be appended")
                return f"data:image/png;base64,{image_path}"
        
        elif isinstance(image_path, bytes):
            # Convert bytes to base64 data URL
            # print("IT IS bytes")
            img_str = base64.b64encode(image_path).decode()
            return f"data:image/png;base64,{img_str}"
        
        else:
            raise ValueError(f"❌ Unsupported image input type: {type(image_path)}")
    
    def predict(self, context, model_input, params=None) -> list:
        """
        Generate captions for input images.
        
        Args:
            context: MLflow context
            model_input: Can be a pandas DataFrame with image column, 
                        a list of images, or a single image
            params: Optional parameters dict with:
                   - max_length: Maximum caption length (default: 128)
                   - text: Optional conditioning text for guided captioning 
                          (default: "Describe this image.")
                   - top_p: Top-p sampling parameter (default: 0.8)
                   - top_k: Top-k sampling parameter (default: 20)
                   - temperature: Temperature for sampling (default: 0.7)
                   - repetition_penalty: Repetition penalty (default: 1.0)
                   
        Returns:
            List of captions
        """
        if params is None:
            params = {}
        
        max_length = params.get('max_length', 256)
        conditional_text = params.get('text', "Describe this image.")
        top_p = params.get('top_p', 0.8)
        top_k = params.get('top_k', 20)
        temperature = params.get('temperature', 0.7)
        repetition_penalty = params.get('repetition_penalty', 1.0)
        
        # Extract image_path from model_input
        model_input = model_input['image_path']
        
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            images = model_input.tolist()[0]
        elif isinstance(model_input, pd.Series):
            images = model_input.tolist()[0]
        elif isinstance(model_input, list):
            images = model_input
        else:
            # Single image
            images = [model_input]
        
        # print("IMAGES:", images)
        # Convert all images to Qwen-compatible format
        image_urls = [self._convert_image(img) for img in images]
        
        # Generate captions
        captions = []
        
        for image_url in image_urls:
            # Construct message for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_url,
                        },
                        {"type": "text", "text": conditional_text},
                    ],
                }
            ]
            
            # Prepare for inference
            # print("MESSAGES:", messages)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate caption
            # print("GENERATING CAPTIONS")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    do_sample=True if temperature > 0 else False
                )
            
            # Decode the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            captions.append(output_text[0])
        
        return captions

# COMMAND ----------

qwen = Qwen3CaptioningModel()

class ContextObject():
  def __init__(self, artifacts):
    self.artifacts = artifacts

artifacts = {}
qwen_context = ContextObject(artifacts)

qwen.load_context(context = qwen_context)

# COMMAND ----------

import os
from PIL import Image

image_dir = "/Volumes/pubsec_video/cv/images"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
pil_images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in image_files]

model_input = {
  "image_path": pil_images + pil_images + pil_images
}

# COMMAND ----------

print(pil_images)

# COMMAND ----------

import timeit
import base64

starting_time = timeit.default_timer()
response = qwen.predict(
  context = None,
  model_input = model_input
)
# print(response.iloc[0].caption)
print(response)
print(f"Inference time: {round((timeit.default_timer() - starting_time)*1000)} ms")

# COMMAND ----------

import timeit
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

# Example usage
image_path = "/Volumes/pubsec_video/cv/images/bruno.png"
image_path = image_to_base64(image_path)
print(type(image_path))

model_input = {
  "image_path": [image_path]
}

starting_time = timeit.default_timer()
response = qwen.predict(
  context = None,
  model_input = model_input
)
# print(response.iloc[0].caption)
print(response)
print(f"Inference time: {round((timeit.default_timer() - starting_time)*1000)} ms")

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# specify the location the model will be saved/registered in Unity Catalog
catalog = "pubsec_video"
schema = "cv"
model_name = "transformers-qwen3-2B-vision"
model_full_name = f"{catalog}.{schema}.{model_name}"
# mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(model_input=model_input, model_output=response)

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
                # 'transformers>=4.30.0',
                'git+https://github.com/huggingface/transformers.git'
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
    'name': 'blip_env'
}

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Qwen3CaptioningModel(),
        signature=signature,
        input_example=model_input,
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

model_uri = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

import timeit

starting_time = timeit.default_timer()
model_output = loaded_model.predict(model_input)
print(f"Inference time: {round((timeit.default_timer() - starting_time)*1000)}ms")
print(model_output)

# COMMAND ----------

import timeit
from PIL import Image
from io import BytesIO
import base64

def pil_to_base64_str(img: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL image to a Base64-encoded string.
    
    Args:
        img: PIL.Image.Image
        format: image format, e.g., 'PNG' or 'JPEG'
        
    Returns:
        str: Base64 string that can be safely passed in JSON
    """
    buf = BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return b64_str

written_images_base64 = []
for pil_img in pil_images + pil_images + pil_images:
  written_images_base64.append(pil_to_base64_str(pil_img))

model_input = {
  "image_path": written_images_base64
}

starting_time = timeit.default_timer()
model_output = loaded_model.predict(model_input)
print(f"Inference time: {round((timeit.default_timer() - starting_time)*1000)}ms")
print(model_output)

# COMMAND ----------

# register the model using the "run" from above.
mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_full_name)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

