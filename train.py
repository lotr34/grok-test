from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import torch
import json

# Model adı
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Tokenizer ve Modeli yükle
try:
    # Force download, en son sürümü almak için
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, force_download=True)
    tokenizer.pad_token = tokenizer.eos_token  # Pad token'ı EOS olarak ayarla

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",  # GPU'yu otomatik seç
        force_download=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id  # Pad token'ı modelin config'ine ayarla
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# LoRA yapılandırması
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Veri setini yükle
try:
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    dataset = Dataset.from_list(lines)
except Exception as e:
    print(f"Veri seti yüklenirken hata oluştu: {e}")
    exit()

# Tokenizasyon fonksiyonu
def tokenize_function(examples):
    try:
        inputs = [msg[0]["content"] for msg in examples["messages"]]
        outputs = [msg[1]["content"] for msg in examples["messages"]]

        # Tokenizer ile inputları ve outputları tokenleştir
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

# Veri setini tokenleştir
try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
except Exception as e:
    print(f"Veri seti tokenleştirilirken hata oluştu: {e}")
    exit()

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=100,
    fp16=False,
    remove_unused_columns=False
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