import time
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from rembg import remove
import os

router = APIRouter()

templates = Jinja2Templates(directory="templates")

@router.post("/remove-background", response_class=HTMLResponse)
async def remove_background(request: Request, image: UploadFile = File(...)):
    image_data = await image.read()
    output = remove(image_data)

    timestamp = int(time.time())
    result_filename = f"processed_image_{timestamp}.png"
    result_path = f"static/processed_images/{result_filename}"
    
    os.makedirs("static/processed_images", exist_ok=True)
    
    with open(result_path, 'wb') as f:
        f.write(output)

    return templates.TemplateResponse('index.html', {
        'request': request, 
        'section': 'background', 
        'image_path_2': result_path,
        'values': {'prompt': ''}
    })
