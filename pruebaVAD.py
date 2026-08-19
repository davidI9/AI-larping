import sounddevice as sd
import numpy as np
import onnxruntime as ort

# 1. Configuración de Audio para Silero
FRECUENCIA = 16000
CHUNK_SIZE = 512 
CANALES = 1
UMBRAL = 0.5  

# 2. Cargar Silero VAD V5 (ONNX)
ruta_modelo = "/home/david/models/silero_vad.onnx"
print(f"Cargando el Oído Interno desde: {ruta_modelo}")
sesion_ort = ort.InferenceSession(ruta_modelo)
print("¡Silero V5 operativo!")

# 3. NUEVO: Memoria a corto plazo de la IA (Unificado en 'state' para la V5)
# El tamaño ahora es (2, batch_size, 128)
estado_interno = np.zeros((2, 1, 128), dtype=np.float32)

estado_actual = "silencio"

# 4. El Interceptor
def callback_microfono(indata, frames, time, status):
    global estado_actual, estado_interno
    
    if status:
        print(f"Aviso de audio: {status}")
        
    audio_chunk = indata.flatten().astype(np.float32)
    audio_tensor = np.expand_dims(audio_chunk, axis=0)
    
    # NUEVO: Inyectamos el audio, la frecuencia y el nuevo 'state'
    entradas = {
        'input': audio_tensor,
        'sr': np.array([FRECUENCIA], dtype=np.int64),
        'state': estado_interno
    }
    
    # Ejecutamos la inferencia
    salidas = sesion_ort.run(None, entradas)
    
    # Extraemos la probabilidad
    probabilidad = salidas[0][0][0]
    
    # NUEVO: Actualizamos el estado interno con la salida del modelo
    estado_interno = salidas[1] 
    
    # 5. La Lógica del Gatillo
    if probabilidad > UMBRAL:
        if estado_actual == "silencio":
            print("\n[🎙️ VOZ DETECTADA - ABRIENDO MICRÓFONO...]")
            estado_actual = "hablando"
    else:
        if estado_actual == "hablando":
            print("[❌ SILENCIO - CERRANDO COMPUERTAS]")
            estado_actual = "silencio"

# 6. Lanzamos el servidor
print("Iniciando captura analógica...")
try:
    with sd.InputStream(device=27, samplerate=FRECUENCIA, channels=CANALES, blocksize=CHUNK_SIZE, dtype=np.float32, callback=callback_microfono):
        print("Habla con normalidad. Pulsa Ctrl+C para apagar.")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nApagando sistema auditivo.")