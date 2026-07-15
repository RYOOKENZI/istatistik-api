from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import comparisons, normality  # normality modülünü içeri aktardık

app = FastAPI(title="İstatistik Analiz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modülleri yayınlıyoruz
app.include_router(comparisons.router, prefix="/api/comparisons", tags=["Karşılaştırmalar"])
app.include_router(normality.router, prefix="/api/normality", tags=["Normallik Testleri"]) # Shapiro-Wilk burada çalışacak

@app.get("/")
def ana_sayfa():
    return {"mesaj": "API Sistemine Hoş Geldiniz. Motor Aktif."}