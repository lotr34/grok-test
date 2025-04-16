import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# spaCy modelini yükle (girdi analizi için)
nlp = spacy.load("en_core_web_sm")

# Mistral modelini ve tokenizer'ı yükle
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Kullanıcı girdisini analiz et
def analiz_et(girdi):
    doc = nlp(girdi)
    temalar = []
    oyunlar = []
    for ent in doc.ents:
        if ent.label_ in ["WORK_OF_ART", "PERSON"]:  # Tema veya oyun adı
            temalar.append(ent.text)
    for token in doc:
        if token.text.lower() in ["monopoly", "risk", "secret hitler"]:  # Oyunlar
            oyunlar.append(token.text)
    return temalar, oyunlar

# Oyun tasarımı için prompt oluştur
def prompt_olustur(kullanici_girdisi, temalar, oyunlar):
    return f"""
    Kullanıcı girdisi: "{kullanici_girdisi}"
    Temalar: {', '.join(temalar)}
    Beğenilen oyunlar: {', '.join(oyunlar)}
    
    Görev: Kullanıcının ilgi alanlarına uygun, profesyonelce tasarlanmış bir kutu oyunu öner. Çıktı şu formatta olsun:
    **Oyun Adı:** [Ad]
    **Tema:** [Tema]
    **Oyuncu Sayısı:** [Sayı]
    **Oynanış Süresi:** [Süre]
    **Mekanikler:** [Mekanikler]
    **Hikaye:** [Hikaye]
    **Kurallar:** [Kurallar]
    **Bileşenler:** [Bileşenler]
    """

# Mistral ile oyun tasarımı üret
def oyun_tasarimi_uret(kullanici_girdisi):
    # Girdiyi analiz et
    temalar, oyunlar = analiz_et(kullanici_girdisi)
    
    # Prompt oluştur
    prompt = prompt_olustur(kullanici_girdisi, temalar, oyunlar)
    
    # Model girdisini tokenize et
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modelden çıktı üret
    outputs = model.generate(
        **inputs,
        max_length=1000,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Çıktıyı decode et
    tasarim = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return tasarim

# Test
kullanici_girdisi = "Yüzüklerin Efendisi evrenini büyüleyici buluyorum. Monopoly ve Risk gibi kutu oyunlarını daha önce oynadım ve keyifli buluyorum. Secret Hitler gibi oyunlarda ilgimi çekmiştir."
print(oyun_tasarimi_uret(kullanici_girdisi))