---
title: Turk Yemek Asistani
emoji: 👁
colorFrom: pink
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Türk yemek kültürünü dijital dünyaya taşıyan yapay zekâ dest
---

# Türk Kültürel Yemek Asistanı 🍲  
### “Türk mutfağının kültürel dokusunu koruyan büyüklerimizin tarifleriyle yapay zekâ destekli dijital tarif asistanı”

---

## 📖 İçindekiler
- [🎯 Proje Amacı](#-proje-amacı)
- [📊 Veri Seti Hakkında](#-veri-seti-hakkında)
- [🧠 Kullanılan Yöntemler ve Teknolojiler](#-kullanılan-yöntemler-ve-teknolojiler)
- [⚙️ Çözüm Mimarisi](#️-çözüm-mimarisi)
- [🌐 Web Arayüzü ve Kullanım Rehberi](#-web-arayüzü-ve-kullanım-rehberi)
- [💻 Kurulum ve Çalıştırma Kılavuzu](#-kurulum-ve-çalıştırma-kılavuzu)
- [✨ Özellikler](#-özellikler)
- [📈 Elde Edilen Sonuçlar](#-elde-edilen-sonuçlar)
  
---

## 🎯 Proje Amacı  

Bu proje, **Türk mutfağının kültürel zenginliğini korumak** ve dijital ortamda yaşatmak amacıyla geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** tabanlı yemek asistanıdır.  

Sistem, kullanıcıdan aldığı yemek adını analiz eder:  
- Eğer tarif **veri setinde bulunuyorsa**, model **hiçbir şekilde metne müdahale etmeden**, yalnızca **biçimsel düzenleme** yapar.  
- Eğer tarif **veri setinde yoksa**, **Gemini API** aracılığıyla Türk mutfağına uygun yeni bir tarif oluşturur.  

🎯 **Hedef:**  
Hem **geleneksel tarifleri korumak**, hem de **kullanıcıların her yemeğe ulaşabileceği bir akıllı tarif sistemi** kurmak.

---

## 📊 Veri Seti Hakkında  

**Kaynak:** [mertbozkurt/turkish-recipe](https://huggingface.co/datasets/mertbozkurt/turkish-recipe)  
**Dosya:** `datav3.txt`  

Veri seti, Türk mutfağından veya diğer mutfaklardan uyarlanmış yüzlerce geleneksel yemeğin detaylı açıklamalarını içerir.  
Her tarif, şu bölümlerden oluşur:
- Başlık (yemek adı)  
- Malzemeler  
- Yapılış adımları  

Uygulama, veri setini otomatik olarak indirir (eğer yerelde yoksa) ve her tarifi ayrı bir belge olarak **LangChain Document** formatında işler.  
Ardından bu belgeler, **Google Embeddings (text-embedding-004)** modeliyle vektörleştirilip **Chroma DB** üzerinde saklanır.  

📌 Böylece kullanıcı sorgusu geldiğinde, sistem **doğrudan ilgili tarifin orijinal metnini bulur** ve hiçbir içerik değişikliğine izin vermez.  

---

## 🧠 Kullanılan Yöntemler ve Teknolojiler  

| Kategori | Teknoloji / Kütüphane | Açıklama |
|-----------|------------------------|-----------|
| 💬 LLM | **Gemini 2.5 Flash (Google Generative AI)** | Tariflerin biçimlendirilmesi ve yeni tarif oluşturma |
| 🧩 Embedding | **Google GenerativeAIEmbeddings (text-embedding-004)** | Tariflerin semantik temsilini oluşturur |
| 🧮 Framework | **LangChain** | RAG pipeline kurulumunda kullanılmıştır |
| 🧠 Veri Tabanı | **Chroma DB** | Tariflerin vektör indekslenmesi ve aranabilirliği |
| 🌐 UI | **Gradio** | Kullanıcı etkileşimi için görsel arayüz |
| ☁️ Deploy | **Hugging Face Spaces** | Uygulamanın bulutta çalıştırıldığı ortam |
| 🔑 Ortam Değişkeni | **dotenv** | API anahtarlarının güvenli yönetimi |
| 🔗 Veri Alma | **requests** | Hugging Face veri setinin otomatik indirilmesi |

---

## ⚙️ Çözüm Mimarisi  

### 🧩 Adım Adım RAG Akışı:
1. **Kullanıcı sorgusu alınır.**
2. Sorgu normalize edilir (ör. “tarifi nedir?” → “nasıl yapılır?”).
3. **Chroma DB retriever** üzerinden en yakın tarif belgeleri getirilir.
4. Eğer sonuç varsa:
   - Orijinal metin korunur.
   - Gemini modeli sadece biçimsel düzenleme yapar.
5. Eğer sonuç yoksa:
   - Gemini modeli sıfırdan yeni bir tarif oluşturur.
6. Sonuç, Gradio arayüzünde biçimlendirilmiş olarak gösterilir.

### 🧠 Mimari Diyagram (özet)

```text
🔹 Kullanıcı Sorgusu
        │
        ▼
  [Retriever - Chroma DB]
        │
        ├── Veri Setinde Tarif Bulunduysa:
        │       ↳ Orijinal Metin + Biçimsel Düzenleme (Gemini)
        │
        └── Tarif Bulunamadıysa:
                ↳ Yeni Tarif Üretimi (Gemini API)
        │
        ▼
  [Gradio Web Arayüzü]
        │
        ▼
  Kullanıcıya Biçimlendirilmiş Sonuç Gösterimi
```
### 🌐 Web Arayüzü ve Kullanım Rehberi

📍 Canlı Demo:
👉 Türk Kültürel Yemek Asistanı - Hugging Face Spaces
[![Hugging Face Space](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-orange?logo=huggingface)](https://huggingface.co/spaces/glcClk/turk-yemek-asistani)

## 📘 Örnek Kullanım 
Kullanım Adımları:
“Hadi bana çocukluğundan belki de özlediğin bir yemek söyle” kutusuna bir yemek adı yazın.
Örnek: Hatay Kağıt Kebabı, Yeşil Mercimekli Semizotu Yemeği, İç Pilavlı Kaburga Dolması, Sodalı Köfte
Ardından dilerseniz sonuna " nasıl yapılır?" ekleyin.

“Tarifi Hazırla” butonuna tıklayın.

Sistem:

Veri setinde varsa: Orijinal Türk tarifi biçimlendirir.

Yoksa: AI (Gemini) tarafından kültürel uyumlu tarif oluşturur.

“Temizle” butonuyla yeni sorgu başlatabilirsiniz.

Eğer bir gecikme yaşanırsa lütfen tekrar "Tarifi Hazırla" butonuna tıklayın.

### 💻 Kurulum ve Çalıştırma Kılavuzu

# 1️⃣ Gerekli kütüphanelerin kurulumu
git clone https://github.com/glcClk/turk-yemek-asistani.git
cd turk-yemek-asistani
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2️⃣ Ortam değişkeni oluşturma
# Proje dizinine .env dosyası oluşturun ve içine:
# GOOGLE_API_KEY=your_api_key_here

# 3️⃣ Uygulamayı başlatma
python app.py

# 4️⃣ Çalışma sonrası
# Gradio arayüzü otomatik olarak tarayıcıda açılacaktır.


### 📸 Ekran Görüntüleri
Lütfen dosyalar bölümüne bakınız.


### ✨ Özellikler

✅ RAG mimarisi ile gerçek tarif arama
✅ Veri setindeki tarifleri kesinlikle değiştirmeme ilkesi
✅ Gemini API ile biçimlendirme ve yeni tarif üretimi
✅ Gradio tabanlı modern web arayüzü, yemekler içeren arka background.jpg entegresi
✅ Hugging Face üzerinde deploy edilmiş canlı sürüm
✅ Otomatik veri indirme ve Chroma DB oluşturma


### 📈 Elde Edilen Sonuçlar

*Veri setinden 3000+ Türk tarifi başarıyla ayrıştırılmıştır.

*Chroma DB hızlı ve doğru benzerlik sonuçları döndürmektedir.

*Gemini Flash modeli ile ortalama yanıt süresi kısaltılmıştır.

*Türk mutfağının özgün yapısı korunarak dijital erişim sağlanmıştır.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
