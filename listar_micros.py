import sounddevice as sd

print("=== DISPOSITIVOS DE AUDIO DISPONIBLES ===")
# Listar todos los dispositivos
for i, d in enumerate(sd.query_devices()):
    # Filtrar solo los que tienen canales de entrada (micrófonos)
    if d['max_input_channels'] > 0:
        print(f"ID [{i}]: {d['name']}")
print("=========================================")