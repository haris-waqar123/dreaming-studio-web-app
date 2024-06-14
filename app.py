from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes import router as main_router
from image_generation import router as image_generation_router
from background_removal import router as background_removal_router
# from voice_cloning import router as voice_cloning_router
# from text_to_audio_gen import router as text_to_audio_gen_router

app = FastAPI()

# Static files setup
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers from different files
app.include_router(main_router)
app.include_router(image_generation_router)
app.include_router(background_removal_router)
# app.include_router(voice_cloning_router)
# app.include_router(text_to_audio_gen_router)
