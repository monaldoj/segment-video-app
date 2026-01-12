# Databricks notebook source
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from PIL import Image
import pandas as pd
from typing import Union, List
import io
import base64
from io import BytesIO
import requests
from databricks.sdk import WorkspaceClient
from openai import OpenAI

class GeminiFlashModelCaptioning(PythonModel):
    """
    MLflow PyFunc wrapper for Databricks Foundation Model API (e.g., Gemini).
    
    This model accepts images in multiple formats:
    - PIL Image objects
    - File paths (strings)
    - Base64 encoded strings
    - Bytes
    - HTTP URLs
    
    Maintains the same input/output signature as BLIPCaptioningModel.
    """
    
    def load_context(self, context):
        """
        Initialize the Databricks client and endpoint configuration.
        
        Args:
            context: MLflow context containing artifacts and parameters
        """
        # Initialize Databricks workspace client
        self.w = WorkspaceClient()
        self.client = self.w.serving_endpoints.get_open_ai_client()
        
        # Default model endpoint (can be overridden in predict params)
        self.default_model_endpoint = "databricks-gemini-3-flash"
        # self.default_model_endpoint = "databricks-gemini-2-5-flash"
        
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
    
    def predict(self, context, model_input, params=None) -> list[str]:
        """
        Generate captions for input images using Databricks Foundation Model API.
        
        Args:
            context: MLflow context
            model_input: Can be a pandas DataFrame with image column, 
                        a list of images, or a single image
            params: Optional parameters dict with:
                   - model_endpoint: Model endpoint name (default: databricks-gemini-3-flash)
                   - text: Optional conditioning text for guided captioning
                   - max_tokens: Maximum tokens to generate
                   - temperature: Sampling temperature
                   
        Returns:
            List of caption strings
        """
        if params is None:
            params = {}
            
        model_endpoint = params.get('model_endpoint', self.default_model_endpoint)
        conditional_text = params.get('text', None)
        max_tokens = params.get('max_tokens', None)
        temperature = params.get('temperature', None)
        
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
        
        # Load all images
        image_urls = [self._convert_image(img) for img in images]
        
        # Generate captions
        captions = []
        for image_url in image_urls:
            # Create prompt
            if conditional_text:
                prompt_text = conditional_text
            else:
                prompt_text = "Describe this image."

            # Prepare content for API
            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
            
            # Prepare API call parameters
            api_params = {
                "model": model_endpoint,
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "reasoning_effort": "low"
            }
            
            # Add optional parameters if provided
            if max_tokens is not None:
                api_params["max_tokens"] = max_tokens
            if temperature is not None:
                api_params["temperature"] = temperature
            
            # Call the API
            response = self.client.chat.completions.create(**api_params)
            
            # Extract caption from response
            caption = response.choices[0].message.content
            captions.append(caption)
        
        return captions

# COMMAND ----------

# from pyspark.sql import functions as F
# import cv2

# # Extract frames from video stored in UC Volumes
# def extract_frames(video_path, num_frames=10):
#     """Extract N frames from video"""
#     # Implementation using cv2 or similar
#     pass

# # Process frames using ai_query
# result_df = spark.sql("""
#     SELECT 
#         video_path,
#         ai_query(
#             'databricks-gemini-3-flash',
#             'Describe what you see in these video frames',
#             files => frame_content
#         ) as description
#     FROM video_frames_table
# """)

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from PIL import Image
import pandas as pd
from typing import Union, List
import io
import json
import base64
from io import BytesIO
import requests
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from pyspark.sql import functions as F
from pyspark.sql.functions import expr
from pyspark.sql.types import StringType, BinaryType
from pyspark.sql.functions import pandas_udf

class GeminiFlashModelCaptioning(PythonModel):
    """
    MLflow PyFunc wrapper for Databricks Foundation Model API (e.g., Gemini).
    
    This model accepts images in multiple formats:
    - PIL Image objects
    - File paths (strings)
    - Base64 encoded strings
    - Bytes
    - HTTP URLs
    
    Maintains the same input/output signature as BLIPCaptioningModel.
    """
    
    def load_context(self, context):
        """
        Initialize the Databricks client and endpoint configuration.
        
        Args:
            context: MLflow context containing artifacts and parameters
        """
        # Initialize Databricks workspace client
        self.w = WorkspaceClient()
        self.client = self.w.serving_endpoints.get_open_ai_client()
        
        # Default model endpoint (can be overridden in predict params)
        self.default_model_endpoint = "databricks-gemini-3-flash"
        
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
    
    def predict(self, context, model_input, params=None) -> list[str]:
        """
        Generate captions for input images using Databricks Foundation Model API.
        
        Args:
            context: MLflow context
            model_input: Can be a pandas DataFrame with image column, 
                        a list of images, or a single image
            params: Optional parameters dict with:
                   - model_endpoint: Model endpoint name (default: databricks-gemini-3-flash)
                   - text: Optional conditioning text for guided captioning
                   - max_tokens: Maximum tokens to generate
                   - temperature: Sampling temperature
                   
        Returns:
            List of caption strings
        """
        if params is None:
            params = {}
            
        model_endpoint = params.get('model_endpoint', self.default_model_endpoint)
        conditional_text = params.get('text', None)
        max_tokens = params.get('max_tokens', None)
        temperature = params.get('temperature', None)
        
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
        
        # Load all images
        image_urls = [self._convert_image(img) for img in images]
        prompt_text = conditional_text if conditional_text else "Describe this image."
        prompts = []
        msgs = []

        for image_url in image_urls:
            # Create prompt
            if conditional_text:
                prompt_text = conditional_text
            else:
                prompt_text = "Describe this image."

            # Prepare content for API
            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]

            # Prepare API call parameters
            api_params = {
                "model": model_endpoint,
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "reasoning_effort": "low"
            }

            # Build named_struct for ai_query from api_params['messages'][0]
            msg = api_params['messages'][0]
            role = msg['role']
            content = msg['content']

            content_expr = (
                f"array("
                f"named_struct('type', 'text', 'text', '{prompt_text}'),"
                f"named_struct('type', 'image_url', 'image_url', named_struct('url', '{image_url}'))"
                f")"
            )

            named_struct_expr = f"named_struct('role', '{role}', 'content', {content_expr})"
            msgs.append(named_struct_expr)

        print(msgs)
        df = spark.createDataFrame([(msg,) for msg in msgs], ["msg"])
        print(df)
        query = f"""
                        ai_query(
                            "{model_endpoint}",
                            request => named_struct("messages",
                                ARRAY({msgs[0]}))
                        )
            """
        print(query)
        # return(query)
        df_with_captions = df.withColumn(
            "caption",
            F.expr(query),    
        )

        # final_df = df_with_captions.withColumn(
        #     "caption", 
        #     F.col("response.choices")[0]["message"]["content"]
        # )

        # Collect captions as a list
        captions = [row.caption for row in df_with_captions.select("caption").collect()]
        
        return captions

# COMMAND ----------

flash = GeminiFlashModelCaptioning()

class ContextObject():
  def __init__(self, artifacts):
    self.artifacts = artifacts

artifacts = {}
flash_context = ContextObject(artifacts)

flash.load_context(context = flash_context)

# COMMAND ----------

import os
from PIL import Image

image_dir = "/Volumes/pubsec_video_processing/cv/images"
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
response = flash.predict(
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
image_path = "/Volumes/pubsec_video_processing/cv/images/bruno.png"
image_path = image_to_base64(image_path)
print(type(image_path))

model_input = {
  "image_path": [image_path]*5
}

starting_time = timeit.default_timer()
response = flash.predict(
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
catalog = "pubsec_video_processing"
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

