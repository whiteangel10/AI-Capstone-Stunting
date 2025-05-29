from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import stunting, mpasi, chatbot

app = FastAPI(title="Stunting")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefixes
app.include_router(stunting.router, prefix="/stunting")
app.include_router(mpasi.router, prefix="/mpasi")
app.include_router(chatbot.router, prefix="/chatbot")
