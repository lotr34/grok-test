import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# spaCy modelini yükle
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"spaCy modeli yüklenirken hata oluştu: {e}")
    exit()

# Eğitilmiş modeli yükle
model_path = "./mistral-finetuned"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Pad token'ı ayarla
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    # Modelin pad_token_id'sini güncelle
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# Kullanıcı girdisini analiz et
def analiz_et(girdi):
    doc = nlp(girdi)
    temalar = []
    oyunlar = []
    for ent in doc.ents:
        if ent.label_ in ["WORK_OF_ART", "LOC", "NORP"]:  # Tema algılama
            temalar.append(ent.text)
    # Oyun isimlerini manuel kontrol et
    oyun_listesi = ["monopoly", "risk", "secret hitler", "carcassonne", "twilight imperium", "terraforming mars", "scythe"]
    for token in doc:
        if token.text.lower() in oyun_listesi:
            oyunlar.append(token.text)
    # Oyun isimlerinin tema olarak algılanmasını önle
    temalar = [t for t in temalar if t.lower() not in oyun_listesi]
    return temalar, oyunlar

# Optimize edilmiş prompt
def prompt_olustur(kullanici_girdisi, temalar, oyunlar):
    return f"""
    Kullanıcı girdisi: "{kullanici_girdisi}"
    Temalar: {', '.join(temalar) if temalar else 'Belirtilen temaya sadık kal'}
    Beğenilen oyunlar: {', '.join(oyunlar) if oyunlar else 'Yok'}

    Görev: Kullanıcının ilgi alanlarına uygun, özgün ve profesyonelce tasarlanmış bir kutu oyunu öner. Oyun, mevcut oyunların kopyası olmamalı ve belirtilen temaya (ör. {', '.join(temalar) if temalar else 'kullanıcı girdisindeki temaya'}) sadık kalmalı. Çıktı şu formatta olsun:
    **Oyun Adı:** [Özgün bir ad]
    **Tema:** [Kullanıcı girdisindeki temaya uygun]
    **Oyuncu Sayısı:** [2-8 arası bir sayı]
    **Oynanış Süresi:** [30-180 dakika]
    **Mekanikler:** [En az 3 mekanik, ör. açık artırma, kart çekme, kaynak toplama]
    **Hikaye:** [Temaya uygun, 50-100 kelimelik etkileyici bir hikaye]
    **Kurallar:** [Adım adım, en az 5 kural]
    **Bileşenler:** [Tahta, kartlar vb., en az 4 bileşen]
    """

# Oyun tasarımı üret
def oyun_tasarimi_uret(kullanici_girdisi):
    try:
        temalar, oyunlar = analiz_et(kullanici_girdisi)
        prompt = prompt_olustur(kullanici_girdisi, temalar, oyunlar)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        outputs = model.generate(
            **inputs,
            max_length=1500,  # Daha uzun çıktılar için
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        tasarim = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return tasarim
    except Exception as e:
        return f"Çıktı üretilirken hata oluştu: {e}"

# Kullanıcı arayüzü
if __name__ == "__main__":
    print("Kutu Oyunu Tasarım Aracına Hoş Geldiniz!")
    print("İlgi alanlarınızı ve sevdiğiniz oyunları yazın (Çıkmak için 'q' yazın):")
    while True:
        kullanici_girdisi = input("> ")
        if kullanici_girdisi.lower() == "q":
            print("Programdan çıkılıyor...")
            break
        print("\nTasarım üretiliyor...\n")
        sonuc = oyun_tasarimi_uret(kullanici_girdisi)
        print(sonuc)
        print("\nBaşka bir oyun tasarlamak için yeni bir girdi yazın:\n")