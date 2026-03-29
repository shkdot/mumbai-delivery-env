import json

with open("signal_matrix.json") as f:
    data = json.load(f)

s = data["signals"]

print("Sample signal counts:")
print(f'Andheri_Station → Powai_Lake: {s["Andheri_Station"]["Powai_Lake"]}')
print(f'NESCO_Warehouse → Goregaon_East: {s["NESCO_Warehouse"]["Goregaon_East"]}')
print(f'Andheri_Station → Jogeshwari_West: {s["Andheri_Station"]["Jogeshwari_West"]}')
print(f'Versova_Beach → IIT_Bombay: {s["Versova_Beach"]["IIT_Bombay"]}')
print(f'Bangur_Nagar → Inorbit_Mall_Malad: {s["Bangur_Nagar"]["Inorbit_Mall_Malad"]}')
print(f'Lokhandwala_Market → Andheri_Station: {s["Lokhandwala_Market"]["Andheri_Station"]}')