# Python 3.10 sürümünü taban olarak alıyoruz
FROM python:3.10-slim

# Çalışma klasörümüzü belirliyoruz
WORKDIR /app

# Önce gereksinim dosyasını kopyalayıp kütüphaneleri kuruyoruz (SciPy vb. için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi diğer tüm kodlarımızı (main.py ve routers klasörünü) kopyalıyoruz
COPY . .

# 8000 portunu dışarı açıyoruz
EXPOSE 8000

# Motoru çalıştıracak ana komut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
