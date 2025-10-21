%%writefile app.py
# --- GEREKLİ KÜTÜPHANELERİ İÇE AKTAR ---
import os
import re
import streamlit as st
import requests
from dotenv import load_dotenv

# LangChain/Gemini Bağımlılıkları
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- RAG SİSTEMİ KURULUMU (TEK SEFERLİK ÇALIŞTIRILIR) ---

# 1️⃣ API Anahtarı ve LLM Kurulumu
# .env dosyasını yerel olarak yükler (Streamlit Cloud'da çalışmaz, orada Secrets kullanılır)
load_dotenv()
GOOGLE_API_KEY_VALUE = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY_VALUE:
    st.error("❌ GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen Streamlit Secrets veya .env dosyanızı kontrol edin.")
    st.stop()

# LLM ve Embedding modelini başlat
@st.cache_resource
def setup_llm_and_embeddings():
    """LLM ve Embedding modellerini Streamlit'in hafızasında önbelleğe alır."""
    # LLM (Gemini-2.5-flash)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3, # KOD 5'teki 0.3'e uyarlanmıştır.
        convert_system_message_to_human=True
    )
    # Embedding Model (text-embedding-004)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return llm, embeddings

llm, embeddings = setup_llm_and_embeddings()

# 2️⃣ Veri Seti Yükleme ve Parçalama (KOD HÜCRESİ 3)
@st.cache_resource
def load_and_chunk_data():
    """Veri setini indirir ve tariflere ayırır."""
    data_path = "datav3.txt"
    hf_url = "https://huggingface.co/datasets/mertbozkurt/turkish-recipe/resolve/main/datav3.txt"
    
    # Dosya indirme (sadece yoksa)
    if not os.path.exists(data_path):
        st.info("🌐 Veri seti Hugging Face'ten indiriliyor...")
        try:
            r = requests.get(hf_url, stream=True)
            r.raise_for_status() # HTTP hatası varsa istisna fırlat
            with open(data_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Veri seti başarıyla indirildi!")
        except Exception as e:
            st.error(f"❌ Veri seti indirilemedi: {e}")
            st.stop()

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
    
    st.success(f"✅ Toplam {len(documents)} tarif yüklendi ve ayrıştırıldı.")
    return documents

texts = load_and_chunk_data()

# 3️⃣ Vektör Veritabanı (Chroma DB) ve Retriever Kurulumu (KOD HÜCRESİ 4)
# Streamlit uygulaması her başladığında DB'yi yeniden oluşturmak pahalıdır.
# Bu nedenle `st.cache_resource` ile önbelleğe alınır.
@st.cache_resource
def setup_vectorstore(texts, embeddings):
    """Chroma DB'yi oluşturur veya yükler ve Retriever'ı başlatır."""
    db_dir = "db_writeable" # Streamlit Cloud'da bu dizin kalıcı olmaz
    
    # Chroma DB oluştur
    with st.spinner("⏳ Yeni Chroma veritabanı oluşturuluyor..."):
        db = Chroma.from_documents(texts, embeddings, persist_directory=db_dir)
        # db.persist() # Streamlit Cloud'da persist işe yaramaz
        retriever = db.as_retriever(search_kwargs={"k": 12})
        st.success(f"✅ Chroma DB başarıyla oluşturuldu! Toplam belge sayısı: {db._collection.count()}")
    return retriever

retriever = setup_vectorstore(texts, embeddings)

# 4️⃣ RAG Zinciri Kurulumu (KOD HÜCRESİ 5)

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
{context}
"""

prompt = PromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

# 5️⃣ Hibrit RAG Fonksiyonu
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

# --- STREAMLIT ARAYÜZ KODU (KOD HÜCRESİ 6) ---

st.set_page_config(
    page_title="Türk Kültürel Yemek Asistanı",
    page_icon="🍲",
    layout="wide"
)

# 💠 Sayfa Arka Planı ve Stil
st.markdown("""
<style>
/* CSS kodları burada kalabilir, Streamlit bileşenlerini stilize eder. */
.main {
    background: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
h1 {
    text-align: center;
    color: darkred;
}
.stButton button {
    background-color: darkred !important;
    color: white !important;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    font-size: 1em;
}
.stButton button:hover {
    background-color: #a30000 !important;
}
</style>
""", unsafe_allow_html=True)

# 💬 Başlık
st.title("🇹🇷 Türk Kültürel Yemek Asistanı")
st.markdown("""
### 🍽️ Merhaba!
Türk mutfağının benzersiz tatlarını keşfetmeye hazır mısın?  
Aşağıya bir yemek adı yaz, birlikte tarifine bakalım! 👇
""")

# 💡 Kullanıcı girişi
query = st.text_input("Bir yemek adı gir (örnek: Tire Şiş Köfte, Karnabahar Pizza, Mercimek Çorbası):")

# 🍳 Buton
if st.button("Tarifi Getir"):
    if not query.strip():
        st.warning("⚠️ Lütfen bir yemek adı gir.")
    else:
        # Arka plan kodunun çalıştığından emin olmak için bu blok eklendi
        if 'retriever' not in locals() or 'llm' not in locals():
            st.error("⚠️ Sistem bileşenleri tam yüklenemedi. Lütfen sayfayı yenilemeyi deneyin.")
        else:
            with st.spinner("Tarif aranıyor..."):
                try:
                    # pure_rag çağrısı, artık tüm bağımlılıklar app.py içinde tanımlı
                    answer = pure_rag(query)
                    st.success("🍴 Tarif hazır!")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")