import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Model ve tokenizer'ı yükle
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pad token'ı ayarla
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    # Modelin pad_token_id'sini güncelle
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# LoRA yapılandırması
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Mistral için hedef modüller
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Veri setini yükle
try:
    dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
except Exception as e:
    print(f"Veri seti yüklenirken hata oluştu: {e}")
    exit()

# Tokenizasyon fonksiyonu
def tokenize_function(examples):
    inputs = [msg[0]["content"] for msg in examples["messages"]]
    outputs = [msg[1]["content"] for msg in examples["messages"]]
    try:
        model_inputs = tokenizer(
            [f"Kullanıcı girdisi: {inp}\nGörev: Kullanıcının ilgi alanlarına uygun bir kutu oyunu öner." for inp in inputs],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = tokenizer(
            outputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )["input_ids"]
        return model_inputs
    except Exception as e:
        print(f"Tokenizasyon sırasında hata oluştu: {e}")
        return None

try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
except Exception as e:
    print(f"Veri seti tokenleştirilirken hata oluştu: {e}")
    exit()

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,  # Bellek dostu
    gradient_accumulation_steps=4,  # Daha büyük etkili batch boyutu
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True,  # Karışık hassasiyet
    remove_unused_columns=False,
    #evaluation_strategy="no"  # Değerlendirme seti yoksa
)

# Trainer oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Eğitimi başlat
print("Eğitim başlıyor...")
try:
    trainer.train()
except Exception as e:
    print(f"Eğitim sırasında hata oluştu: {e}")
    exit()

# Modeli kaydet
model.save_pretrained("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")
print("Model kaydedildi: ./mistral-finetuned")