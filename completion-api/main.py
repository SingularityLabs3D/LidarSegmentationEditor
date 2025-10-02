import math
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse
import numpy as np
import json
import os
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import open3d as o3d
import tempfile
from starlette.background import BackgroundTask

from diffusion_model import DiffCompletion, load_pcd_from_bytes

load_dotenv()

app = FastAPI()

# Define the Pydantic model for the configuration
class DiffusionConfig(BaseModel):
    denoising_steps: int = Field(20, ge=0, le=50, description="Number of DPM-Solver steps.")
    starting_point: int = Field(..., ge=0, le=999, description="Starting point of the diffusion.")
    cond_weight: float = Field(0.0, ge=0.0, description="Conditioning weight.")
    multiplication_factor: float = Field(..., gt=0, description="Factor to increase the number of points by.")

# Dependency to parse and validate the config JSON from the form
def parse_config(config_json: str = Form(...)) -> DiffusionConfig:
    try:
        config_data = json.loads(config_json)
        return DiffusionConfig(**config_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in config_json.")
    except ValidationError as e:
        # FastAPI can nicely format Pydantic validation errors
        raise HTTPException(status_code=422, detail=e.errors())

def cleanup_file(file_path: str):
    """Remove a file in the background."""
    try:
        os.remove(file_path)
    except OSError:
        pass

@app.post("/diffuse-points/")
async def diffuse_point_cloud(
    file: UploadFile = File(...),
    config: DiffusionConfig = Depends(parse_config)
):
    """
    Accepts a point cloud file (.ply, .pcd) and a JSON configuration string.
    Returns the diffused and refined point cloud as a .pcd file.
    
    The configuration should be provided as a JSON string in the 'config_json' form field.

    Example usage with curl:
    curl -X POST "http://127.0.0.1:8000/diffuse-points/" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/file.ply" -F 'config_json={"starting_point": 999, "multiplication_factor": 1.5}' -o diffused_output.pcd
    """
    if not (file.filename.endswith(".ply") or file.filename.endswith(".pcd")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .ply and .pcd are supported.")
    
    config_dict = config.model_dump()

    # Add model paths from environment variables
    config_dict['diff'] = os.getenv("DIFF_MODEL_PATH")
    config_dict['refine'] = os.getenv("REFINE_MODEL_PATH")

    if not config_dict['diff'] or not os.path.exists(config_dict['diff']):
        raise HTTPException(status_code=500, detail=f"Diffusion model not found at path: {config_dict['diff']}")
    if not config_dict['refine'] or not os.path.exists(config_dict['refine']):
        raise HTTPException(status_code=500, detail=f"Refinement model not found at path: {config_dict['refine']}")

    contents = await file.read()
    
    try:
        points = load_pcd_from_bytes(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read point cloud file: {e}")
    
    # Calculate num_points and duplication_factor based on the input file and multiplication_factor
    num_input_points = points.shape[0]
    config_dict["num_points"] = int(config_dict["multiplication_factor"] * num_input_points)
    config_dict["duplication_factor"] = math.ceil(config_dict["multiplication_factor"])
    config_dict["range"] = "radius"

    diff_completion_model = None
    try:
        # Create the model for this request
        diff_completion_model = DiffCompletion(config_dict, full_exp_dir="exp")
        
        min0 = points[:, 0].min()
        min1 = points[:, 1].min()
        min2 = points[:, 2].min()
        points[:, 0] = points[:, 0] - min0
        points[:, 1] = points[:, 1] - min1
        points[:, 2] = points[:, 2] - min2
        # Perform diffusion
        refine_scan, _ = diff_completion_model.denoise_scan(points)

        refine_scan[:, 0] += min0
        refine_scan[:, 1] += min1
        refine_scan[:, 2] += min2
        
        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(refine_scan)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as tmp:
            temp_path = tmp.name
            o3d.io.write_point_cloud(temp_path, pcd, write_ascii=False)

        # Return the file as a response, with a background task to clean it up
        return FileResponse(
            path=temp_path,
            media_type='application/octet-stream',
            filename='diffused_points.pcd',
            background=BackgroundTask(cleanup_file, file_path=temp_path)
        )

    except Exception as e:
        # This will catch errors during model loading or processing
        raise HTTPException(status_code=500, detail=f"An error occurred during diffusion processing: {e}")
    finally:
        # Clean up the model and clear CUDA cache
        if diff_completion_model is not None:
            del diff_completion_model
            torch.cuda.empty_cache()
