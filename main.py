from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import normality

app = FastAPI(title="İstatistik Motoru API")

# DÜZELTİLMİŞ CORS AYARI (allow_credentials kesinlikle False olmalı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.include_router(normality.router)

@app.get("/")
async def root():
    # BURASI ÇOK ÖNEMLİ: Tarayıcıda artık bu mesaj görünmeli!
    return {"mesaj": "İstatistik API Motoru Sorunsuz Çalışıyor!"}

@app.get("/health")
async def health():
    return {"status": "ok"}
