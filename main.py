from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import normality
from routers import variance  # Levene vb. testler

app = FastAPI(
    title="StatLabseu API Motoru",
    description="İstatistiksel analizler için veri işleme ve test motoru",
    version="1.0.0"
)

# GÜVENLİ CORS AYARI (Sadece senin sitelerine izin verir)
# Buraya kendi domainlerini ve lokal test portlarını yazıyoruz
origins = [
    "https://statlabseu.com",
    "https://www.statlabseu.com",
    "http://localhost:5500",    # VS Code Live Server gibi araçlar için lokal port
    "http://127.0.0.1:5500",
    "http://localhost:3000",    # React/Vue kullanırsan lokal port
    "http://127.0.0.1:8000"     # Doğrudan API testi için
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # "*" yerine sadece listedeki sitelere izin veriyoruz!
    allow_credentials=True,     # Özel domain belirttiğimiz için artık True yapabiliriz (İleride login sistemi yaparsan cookie'ler için şart)
    allow_methods=["GET", "POST", "OPTIONS"], # İzin verilen HTTP metodlarını sınırla
    allow_headers=["*"],
)

# Router'ları projeye dahil et
app.include_router(normality.router)
app.include_router(variance.router)

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
