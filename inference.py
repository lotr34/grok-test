import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Modelin path'ini belirt (fine-tuned model burada olmalı)
    model_path = "./mistral-finetuned"

    # Tokenizer ve model yükle
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    print("🎲 Kutu Oyunu Tasarım Modeli Aktif!")
    print("Prompt girin (çıkmak için 'q' yazın):")

    while True:
        user_input = input("\n📝 Kullanıcı girdisi: ")
        if user_input.lower() == 'q':
            print("Çıkılıyor...")
            break

        # Prompt'u oluştur (istersen sistem mesajı ya da görev ekleyebilirsin)
        prompt = f"Kullanıcı girdisi: {user_input}\nGörev: Kullanıcının ilgi alanlarına göre özgün bir kutu oyunu öner."

        # Modelden yanıt al
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\n🎮 Önerilen Oyun Tasarımı:\n")
        print(decoded_output.split("Görev:")[-1].strip())

if __name__ == "__main__":
    main()
