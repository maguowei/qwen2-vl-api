from fastapi import FastAPI
from qwen2_vl import extract_info


app = FastAPI()


@app.get("/api/image")
def read_root(image_url: str, prompt: str = '描述一下这张图片', resized_width: int =200, resized_height: int =200):
    result = extract_info(image_url, prompt, resized_width, resized_height)
    return {"data": result}
