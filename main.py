from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import normality

app = FastAPI(title="İstatistik Motoru API")

# Tarayıcı güvenlik duvarını (CORS) aşmak için gerekli ayar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Şimdilik her web sitesinden gelen isteği kabul et
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST vb. hepsine izin ver
    allow_headers=["*"],
)

app.include_router(normality.router)

@app.get("/")
def read_root():
    return {"mesaj": "İstatistik API Motoru Sorunsuz Çalışıyor!"}
