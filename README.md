# PDF Chat Uygulaması - Google Gemini 2.5 Flash



https://github.com/user-attachments/assets/9530fb57-7d27-4fa7-85f2-07c478c584f2





** Video'da görecekleriniz:**
- ✅ PDF yükleme ve analiz
- ✅ Türkçe soru-cevap örnekleri  
- ✅ Tam özet çıkarma
- ✅ Gemini 2.5 Flash performansı

---

Bu proje, PDF dosyalarını yükleyip içeriği hakkında Türkçe sohbet edebileceğiniz bir Streamlit uygulamasıdır. Google Gemini 2.5 Flash, Langchain kütüphanesi ve Google embeddings kullanılarak geliştirilmiştir.

##  Özellikler

- **PDF Yükleme**: PDF dosyalarınızı kolayca yükleyin
- **Türkçe Sohbet**: PDF içeriği hakkında Türkçe sorular sorun
- **Tam Özet**: PDF'in tamamından kapsamlı özet çıkarın
- **Konuşma Geçmişi**: Önceki sorularınızı ve cevapları hatırlar
- **Google Gemini 2.5 Flash**: En son AI teknolojisi
- **Vektör Veritabanı**: ChromaDB ile hızlı arama

##  Gereksinimler

- Python 3.8+
- Google AI Studio API Key ([buradan alın](https://aistudio.google.com/apikey))

## Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/Bariskaya07/LangChain-PDF-Chat-Turkish-with-Gemini.git
cd LangChain-PDF-Chat-Turkish-with-Gemini
```

2. **Virtual environment oluşturun:**
```bash
python3 -m venv pdf_chat_env
source pdf_chat_env/bin/activate  # Linux/Mac
# veya
pdf_chat_env\Scripts\activate     # Windows
```

3. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Uygulamayı başlatın:**
```bash
streamlit run chat_pdf_google.py
```

##  Kullanım

1. **API Key Girişi**: Google AI Studio'dan aldığınız API key'i girin
2. **PDF Yükleme**: PDF dosyanızı sürükle-bırak ile yükleyin
3. **Sohbet**: PDF hakkında Türkçe sorular sorun
4. **Tam Özet**: "📄 PDF'in Tam Özetini Çıkar" butonuna tıklayın

##  Teknik Detaylar

- **AI Modeli**: Google Gemini 2.5 Flash
- **Embedding**: Google Embedding Model (models/embedding-001)
- **Vektör DB**: ChromaDB
- **Text Splitter**: RecursiveCharacterTextSplitter (2000 chunk, 400 overlap)
- **Framework**: Streamlit + LangChain

##  Proje Yapısı

```
pdf-chat-gemini/
├── chat_pdf_google.py      # Ana uygulama
├── requirements.txt        # Python paketleri
├── README.md              # Bu dosya
├── db/                    # Vektör veritabanı (otomatik oluşur)
└── pdf_chat_env/          # Virtual environment
```

##  Öne Çıkan Fonksiyonlar

### Kapsamlı PDF Analizi
- Büyük chunk boyutu (2000 karakter)
- Yüksek overlap (400 karakter) 
- 20'ye kadar chunk'ı aynı anda analiz

### Türkçe Optimizasyonu
- Türkçe sistem prompt'ları
- Türkçe cevap formatı
- Kültürel bağlama uygun yanıtlar

##  Güvenlik

- API key'ler güvenli şekilde saklanır
- Geçici dosyalar otomatik temizlenir
- Kullanıcı verileri yerel olarak işlenir

##  Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin yeni-ozellik`)
5. Pull Request oluşturun

##  Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

##  Destek

Sorun yaşıyorsanız:
- Issues bölümünde sorun bildirin
- Detaylı hata mesajları paylaşın
- Kullandığınız Python versiyonunu belirtin

##  Sistem Gereksinimleri

- **RAM**: Minimum 4GB (8GB önerilir)
- **Disk**: 500MB boş alan
- **İnternet**: API çağrıları için stabil bağlantı

---

 **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**
