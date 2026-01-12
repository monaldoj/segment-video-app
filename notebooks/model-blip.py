# Databricks notebook source
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import pandas as pd
from typing import Union, List
import io
import base64
from io import BytesIO
import requests

class BLIPCaptioningModel(PythonModel):
    """
    MLflow PyFunc wrapper for Salesforce BLIP image captioning model.
    
    This model accepts images in multiple formats:
    - PIL Image objects
    - File paths (strings)
    - Base64 encoded strings
    - Bytes
    """
    
    def load_context(self, context):
        """
        Load the BLIP model and processor.
        
        Args:
            context: MLflow context containing artifacts and parameters
        """
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        model_name = "Salesforce/blip-image-captioning-large"
        # model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    # def _load_image(self, image_input) -> Image.Image:
    #     """
    #     Convert various image input formats to PIL Image.
        
    #     Args:
    #         image_input: Image in various formats (path, bytes, base64, PIL Image)
            
    #     Returns:
    #         PIL Image object
    #     """
    #     if isinstance(image_input, Image.Image):
    #         return image_input
    #     elif isinstance(image_input, str):
    #         # Check if it's a base64 string
    #         if image_input.startswith('data:image'):
    #             # Remove data URL prefix
    #             base64_str = image_input.split(',')[1]
    #             image_bytes = base64.b64decode(base64_str)
    #             return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    #         elif image_input.startswith('base64:'):
    #             # Custom base64 prefix
    #             base64_str = image_input[7:]
    #             image_bytes = base64.b64decode(base64_str)
    #             return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    #         else:
    #             # Assume it's a file path
    #             return Image.open(image_input).convert('RGB')
    #     elif isinstance(image_input, bytes):
    #         return Image.open(io.BytesIO(image_input)).convert('RGB')
    #     else:
    #         raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _convert_image(self, image_path) -> Image.Image:
        if isinstance(image_path, Image.Image):
            print("IT IS AN IMAGE")
            image = image_path
        elif image_path.startswith("http"):
            print("IT IS AN HTTP")
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        elif image_path.startswith('/'):
            print("IT IS A /")
            image = Image.open(image_path).convert("RGB")
        elif image_path.startswith('data:image'):
            print("IT IS A data:image")
            header, b64_string = image_path.split(",", 1)
            image_bytes = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_bytes))
        elif isinstance(image_path, str):
            print("IT IS A str")
            image_bytes = base64.b64decode(image_path)
            image = Image.open(BytesIO(image_bytes))
        elif isinstance(image_path, bytes):
            print("IT IS bytes")
            image = Image.open(BytesIO(image_bytes))
        else:
            # return "❌ Unsupported image type"
            raise ValueError(f"❌ Unsupported image input type: {type(image_path)}")
        return image
    
    def predict(self, context, model_input, params=None) -> list[str]:
        """
        Generate captions for input images.
        
        Args:
            context: MLflow context
            model_input: Can be a pandas DataFrame with image column, 
                        a list of images, or a single image
            params: Optional parameters dict with:
                   - max_length: Maximum caption length (default: 50)
                   - num_beams: Number of beams for beam search (default: 4)
                   - text: Optional conditioning text for guided captioning
                   
        Returns:
            pandas DataFrame with 'caption' column or list of captions
        """
        if params is None:
            params = {}
            
        max_length = params.get('max_length', 50)
        num_beams = params.get('num_beams', 4)
        conditional_text = params.get('text', None)
        model_input = model_input['image_path']
        # print(type(model_input))
        # print("before processing:", model_input)
        
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            # Assume first column contains images
            # print("pandas")
            # print(model_input)
            images = model_input.tolist()[0]
        elif isinstance(model_input, pd.Series):
            images = model_input.tolist()[0]
        elif isinstance(model_input, list):
            # print("list")
            images = model_input
        else:
            # Single image
            # print("single")
            images = [model_input]
        
        # Load all images
        # pil_images = [self._load_image(img) for img in images]
        pil_images = [self._convert_image(img) for img in images]
        
        # Generate captions
        captions = []
        for pil_image in pil_images:
            if conditional_text:
                # Conditional captioning with text prompt
                inputs = self.processor(
                    pil_image, 
                    conditional_text, 
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Unconditional captioning
                inputs = self.processor(
                    pil_image, 
                    return_tensors="pt"
                ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
            
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
            captions.append(caption)
        
        # Return as DataFrame
        return captions
        # return pd.DataFrame({'caption': captions})



# COMMAND ----------

blip = BLIPCaptioningModel()

class ContextObject():
  def __init__(self, artifacts):
    self.artifacts = artifacts

artifacts = {}
blip_context = ContextObject(artifacts)

blip.load_context(context = blip_context)

# COMMAND ----------

import os
from PIL import Image

image_dir = "/Volumes/justinm/cv/images"
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
response = blip.predict(
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
image_path = "/Volumes/justinm/cv/images/bruno.png"
image_path = image_to_base64(image_path)

model_input = {
  "image_path": [image_path]
}

starting_time = timeit.default_timer()
response = blip.predict(
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
catalog = "justinm"
schema = "cv"
model_name = "transformers-blip"
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
        python_model=BLIPCaptioningModel(),
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

