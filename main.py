from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import normality
from routers import variance  # Levene vb. testler
from routers import t_tests   # YENİ EKLENEN: t-Testleri modülü
from routers import z_tests
from routers import wilcoxon_tests

app = FastAPI(
    title="StatLabseu API Motoru",
    description="İstatistiksel analizler için veri işleme ve test motoru",
    version="1.0.0"
)

# GÜVENLİ CORS AYARI (Sadece senin sitelerine izin verir)
origins = [
    "https://statlabseu.com",
    "https://www.statlabseu.com",
    "http://localhost:5500",    # VS Code Live Server
    "http://127.0.0.1:5500",
    "http://localhost:3000",    # React/Vue port
    "http://127.0.0.1:8000"     # API test port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Router'ları projeye dahil et
app.include_router(normality.router)
app.include_router(variance.router)
app.include_router(t_tests.router) # YENİ EKLENEN: t-Testlerini API'ye bağlıyoruz
app.include_router(z_tests.router)
app.include_router(wilcoxon_tests.router)



@app.get("/")
async def root():
    # Tarayıcıda API'nin kök dizinine girildiğinde görünen mesaj
    return {
        "mesaj": "StatLabseu API Motoru Sorunsuz Çalışıyor!",
        "versiyon": "1.0.0",
        "durum": "Aktif"
    }

@app.get("/health")
async def health():
    # Sunucunun ayakta olup olmadığını kontrol eden sağlık uç noktası
    return {"status": "ok"}
