# --- GEREKLİ KÜTÜPHANELERİ İÇE AKTAR ---
import os
import re
import requests
import gradio as gr # Streamlit yerine Gradio eklendi
from dotenv import load_dotenv

# LangChain/Gemini Bağımlılıkları
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- RAG SİSTEMİ KURULUMU (TEK SEFERLİK ÇALIŞTIRILIR) ---

# 1️⃣ API Anahtarı ve LLM Kurulumu
load_dotenv()
GOOGLE_API_KEY_VALUE = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY_VALUE:
    # Streamlit komutu yerine Python Exception kullanıldı.
    raise Exception("❌ GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen Hugging Face Secrets ayarını kontrol edin.")

# LLM ve Embedding modelini başlat
# Streamlit dekoratörleri kaldırıldı, fonksiyonlar standart Python fonksiyonu olarak bırakıldı.
def setup_llm_and_embeddings():
    """LLM ve Embedding modellerini başlatır."""
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3, # KOD 5'teki 0.3'e uyarlanmıştır.
        convert_system_message_to_human=True
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return llm, embeddings

llm, embeddings = setup_llm_and_embeddings()

# 2️⃣ Veri Seti Yükleme ve Parçalama (KOD HÜCRESİ 3)
# Streamlit komutları kaldırıldı.
def load_and_chunk_data():
    """Veri setini indirir ve tariflere ayırır."""
    data_path = "datav3.txt"
    hf_url = "https://huggingface.co/datasets/mertbozkurt/turkish-recipe/resolve/main/datav3.txt"
    
    # Dosya indirme (sadece yoksa)
    if not os.path.exists(data_path):
        print("🌐 Veri seti Hugging Face'ten indiriliyor...") # st.info yerine print
        try:
            r = requests.get(hf_url, stream=True)
            r.raise_for_status() # HTTP hatası varsa istisna fırlat
            with open(data_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Veri seti başarıyla indirildi!") # st.success yerine print
        except Exception as e:
            raise Exception(f"❌ Veri seti indirilemedi: {e}") # st.error ve st.stop yerine raise Exception

    # Dosyayı oku ve tariflere ayır
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    documents = []
    current_recipe = ""
    for line in lines:
        if re.search(r"nasıl yapılır\?", line, re.IGNORECASE):
            if current_recipe.strip():
                documents.append(Document(page_content=current_recipe.strip()))
            current_recipe = line
        else:
            current_recipe += line
    if current_recipe.strip():
        documents.append(Document(page_content=current_recipe.strip()))
        print(f"✅ Toplam {len(documents)} tarif yüklendi ve ayrıştırıldı.") # st.success yerine print
    return documents

texts = load_and_chunk_data()

# 3️⃣ Vektör Veritabanı (Chroma DB) ve Retriever Kurulumu (KOD HÜCRESİ 4)
# Streamlit dekoratörleri ve komutları kaldırıldı.
def setup_vectorstore(_texts, _embeddings): # Hata çözümü için _texts ve _embeddings kullanıldı.
    """Chroma DB'yi oluşturur veya yükler ve Retriever'ı başlatır."""
    db_dir = "db_writeable" 
    
    # Chroma DB oluştur
    print("⏳ Yeni Chroma veritabanı oluşturuluyor...") # st.spinner yerine print
    db = Chroma.from_documents(_texts, _embeddings, persist_directory=db_dir) # Hata çözümü için _texts, _embeddings kullanıldı.
    retriever = db.as_retriever(search_kwargs={"k": 12})
    print(f"✅ Chroma DB başarıyla oluşturuldu! Toplam belge sayısı: {db._collection.count()}") # st.success yerine print
    return retriever

retriever = setup_vectorstore(texts, embeddings)

# 4️⃣ RAG Zinciri Kurulumu (KOD HÜCRESİ 5) - TAMAMEN KORUNMUŞTUR

# Prompt: veri varsa düzenle, yoksa AI tarif üret
template = """Aşağıda yemek veri setinden getirilen 'Bağlam' metni bulunmaktadır.
🟢 Eğer 'Bağlam' boş **değilse**:
Sadece biçimlendir — yeni bilgi ekleme, çıkarma veya değiştirme YASAKTIR.
Bağlamı temiz bir yemek tarifi biçiminde (Başlık, Malzemeler, Yapılışı) sun.
Bu metni düzenli biçimde yaz:
- Başlık, Malzemeler, Yapılışı bölümleri olmalı.
- Orijinal cümleleri kesinlikle koru.
- Metni **madde madde** (1., 2., 3. şeklinde) yaz.
- Yeni bilgi ekleme, çıkarma veya değiştirme YASAKTIR.
🔵 Eğer 'Bağlam' **boşsa**:
Bu yemeğin adıyla uyumlu, mantıklı bir tarif oluştur.
(Başlık, Malzemeler, Yapılışı başlıklarıyla yaz.)
Cümlenin sonunda şunu ekle:
"(Bu tarif AI tarafından oluşturulmuştur.)"
---
Sorgu: {input}
Bağlam:
{context}"""

prompt = PromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

# 5️⃣ Hibrit RAG Fonksiyonu - TAMAMEN KORUNMUŞTUR
def pure_rag(query: str):
    """Veri setinden tarif arar, bulamazsa AI'ya oluşturur."""
    
    # Sorguyu normalize et (KOD 5'teki normalizasyon)
    normalized_query = query.strip().lower()
    normalized_query = re.sub(r"tarifi nedir\??", "nasıl yapılır?", normalized_query)
    normalized_query = re.sub(r"tarifi", "nasıl yapılır", normalized_query)
    
    # Veri tabanında arama
    results = retriever.get_relevant_documents(normalized_query)
    context = "\n\n".join([d.page_content for d in results if d.page_content.strip()])
    
    # Alternatif sorgular (KOD 5'teki yedekleme)
    if not context.strip():
        alt_forms = [
            query,
            query.replace("tarifi", "nasıl yapılır"),
            query.replace("nedir", ""),
            query.replace("tarifi nedir", "tarifini yazar mısın"),
            query.replace("tarifi nedir?", "")
        ]
        for alt in alt_forms:
            alt_results = retriever.get_relevant_documents(alt)
            if alt_results:
                context = "\n\n".join([d.page_content for d in alt_results if d.page_content.strip()])
                if context.strip():
                    break
        
    if not context.strip():
        context = ""
        
    # Gemini’ye gönder
    docs = [Document(page_content=context)]
    response = document_chain.invoke({"input": query, "context": docs})
    return response

# --- GRADIO ARAYÜZÜ (GÖRÜNÜM İYİLEŞTİRMELERİ VE CSS) ---

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Marcellus&display=swap');

/* ============================================================
   1. ARKA PLAN
   ============================================================ */
html, body {
    background: linear-gradient(
        rgba(255, 248, 225, 0.75),
        rgba(255, 248, 225, 0.75)
    ), url('https://huggingface.co/spaces/glcClk/turk-yemek-asistani/resolve/main/background.jpg')
       no-repeat center center fixed !important;
    background-size: cover !important;
    font-family: 'Marcellus', serif !important;
}

/* ============================================================
   2. GENEL METİN STİLİ
   ============================================================ */
h1, p, label, textarea, input {
    color: #2C2A2A !important;
    font-family: 'Marcellus', serif !important;
}
h1 {
    color: #8B0000 !important;
    font-size: 2.8em !important;
    text-shadow: 1px 1px 2px #FAD7A0;
}

/* ============================================================
   3. GİRİŞ KUTUSU (sol)
   ============================================================ */
textarea, input {
    background-color: #FFF8DC !important;
    border: 2px solid #C0392B !important;
    border-radius: 10px !important;
    color: #2C2A2A !important;
}

/* ============================================================
   4. TARİF KUTUSU (sağ)
   ============================================================ */
.output-markdown, [class*="output"] {
    background-color: rgba(213, 232, 184, 0.94) !important;  /* yağ yeşili */
    backdrop-filter: blur(4px);
    border: 3px solid #7D6608 !important;
    border-radius: 15px !important;
    color: #0B1E3D !important;
    padding: 24px !important;
    line-height: 1.7em !important;
    font-size: 1.08em !important;
    font-weight: 500 !important;
    box-shadow: 0 0 15px rgba(0,0,0,0.25);
    overflow-y: auto !important;
}

/* Tüm markdown içerikleri için siyah renk garantisi */
.output-markdown *, [class*="output"] * {
    color: #0B1E3D !important;
    text-shadow: none !important;
}

/* ============================================================
   5. BUTONLAR
   ============================================================ */
button.primary {
    background-color: #C0392B !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}
button.primary:hover {
    background-color: #922B21 !important;
}
button.secondary {
    background-color: #D7CCC8 !important;
    color: #2C2C2C !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}
button.secondary:hover {
    background-color: #BCAAA4 !important;
}
"""

# ... (iface = gr.Interface(...)

iface = gr.Interface(
    fn=pure_rag,
    inputs=gr.Textbox(
        lines=2, 
        label="Hadi bana çocukluğundan belki de özlediğin bir yemek söyle:", 
        placeholder="Örnek: Hatay Kağıt Kebabı, Sodalı Köfte, İç Pilavlı Kaburga Dolması, Yeşil Mercimekli Semizotu Yemeği"
    ),
    outputs=gr.Markdown(label="🍴 Tarif Sonucu"),
    title="Türk Kültürel Yemek Asistanı 🍲", 
    description="Türk mutfağının benzersiz tatlarını keşfetmeye hazır mısın? Aşağıya bir yemek adı yaz, birlikte tarifine bakalım! Ayrıca tabiki bana tüm yemekleri sorabilirsin.",
    theme=gr.themes.Soft(primary_hue="red").set(body_background_fill="#FFF8E1"),
    css=custom_css,
    submit_btn="Tarifi Hazırla", 
    clear_btn="Temizle"
)

if __name__ == "__main__":
    iface.launch(share=False)
    