import json

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

for item in data:
    output = item["messages"][1]["content"]
    if "Hikaye" not in output:
        tema = output.split("Tema: ")[1].split("\n")[0]
        mekanikler = output.split("Mekanikler: ")[1].split("\n")[0]
        output += f"\n**Hikaye:** {tema} temalı bir dünyada, oyuncular stratejik kararlarla zafer için yarışır.\n**Kurallar:** 1. Oyuncular her turda kart çeker. 2. {mekanikler} mekanikleriyle kaynak toplanır. 3. Puanlar inşa edilen yapılarla kazanılır. 4. Özel kartlar avantaj sağlar. 5. 10 tur sonunda en yüksek puan kazanır.\n**Bileşenler:** 1 tahta, 60 kart, 30 minyatür, 20 jeton."
        item["messages"][1]["content"] = output

with open("dataset_updated.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")