from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import normality

# API'mizi başlatıyoruz
app = FastAPI(title="İstatistik Motoru API")

# Tarayıcı güvenlik duvarını (CORS) aşmak için gerekli ayar
# allow_origins=["*"] diyerek tüm web sitelerinden gelen isteklere izin veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST vb. hepsine izin ver
    allow_headers=["*"],
)

# Normallik testlerinin (Shapiro-Wilk) bulunduğu rotayı sisteme bağlıyoruz
app.include_router(normality.router)

# Ana Sayfa (Motorun çalışıp çalışmadığını test etmek için)
@app.get("/")
async def root():
    return {"mesaj": "İstatistik API Motoru Sorunsuz Çalışıyor!"}

# Sağlık Kontrolü (Dokploy gibi sunucuların sistemi kontrol etmesi için)
@app.get("/health")
async def health():
    return {"status": "ok"}

# Kendi bilgisayarında test edebilmen için gerekli blok
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
