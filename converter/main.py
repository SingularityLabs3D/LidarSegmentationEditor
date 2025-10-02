from fastapi import FastAPI, HTTPException
from converter import run_conversion_pipeline
import os
import pathlib

app = FastAPI()

# Hardcoded parent directory for all conversion outputs
print(os.getenv("WORKDIR"), flush=True)
OUTPUT_DIRECTORY = pathlib.Path(os.getenv("WORKDIR"))
OUTPUT_DIRECTORY.mkdir(exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


@app.get("/convert")
async def convert_file(path: str):
    """
    Converts a file using the conversion pipeline.
    Expects a 'path' query parameter with the path to the file.
    Example: http://127.0.0.1:8000/convert?path=C:\\Users\\murka\\Desktop\\converter\\first_chunk_22.pcd
    """
    input_path = pathlib.Path(path)
    if not input_path.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found at: {path}")

    # try:
    # Create a unique output subdirectory for this conversion run
    # e.g., input 'my_file.pcd' -> output 'conversion_output/my_file/'
    file_stem = input_path.stem
    
    timings = run_conversion_pipeline(str(input_path), OUTPUT_DIRECTORY)
    
    return {
        "message": "Conversion successful",
        "output_directory": os.path.abspath(OUTPUT_DIRECTORY),
        "timings": timings
    }
    # except Exception as e:
    #     # Catches errors from the conversion pipeline
    #     raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

