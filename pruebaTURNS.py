import sounddevice as sd
import numpy as np
import onnxruntime as ort
import threading
import queue
from collections import deque
from transformers import AutoProcessor

# 1. Configuración de Audio
FRECUENCIA = 16000
CANALES = 1
MAX_MUESTRAS = FRECUENCIA * 3 

buffer_audio = deque(maxlen=MAX_MUESTRAS)
cola_procesamiento = queue.Queue()

# 2. Cargar el preprocesador y TU modelo ONNX local
print("Cargando procesador acústico...")
# Esto carga solo las reglas matemáticas (pesa unos KBs)
processor = AutoProcessor.from_pretrained("videosdk-live/Namo-Turn-Detector-v1-Multilingual")

ruta_modelo = "/home/david/models/Namo-Turn-Detector-v1-Multilingual.onnx"
print(f"Cargando TU modelo ONNX local desde: {ruta_modelo}")
sesion_ort = ort.InferenceSession(ruta_modelo) 
print("¡Sistema listo!")

# 4. Hilo de captura
def callback_microfono(indata, frames, time, status):
    audio_plano = indata.flatten().astype(np.float32)
    cola_procesamiento.put(audio_plano)

def hilo_captura():
    with sd.InputStream(device=27, samplerate=FRECUENCIA, channels=CANALES, dtype=np.float32, callback=callback_microfono):
        while True:
            sd.sleep(100)

thread_micro = threading.Thread(target=hilo_captura, daemon=True)
thread_micro.start()
print("Micrófono escuchando en segundo plano...")

# 5. El Bucle Principal
try:
    while True:
        nuevo_chunk = cola_procesamiento.get() 
        buffer_audio.extend(nuevo_chunk)
        
        # Evaluamos cuando tenemos suficiente audio
        if len(buffer_audio) > FRECUENCIA: 
            audio_ventana = np.array(buffer_audio, dtype=np.float32)
            
            # 1. El procesador crea el attention_mask y los inputs_ids en formato Numpy ("np")
            inputs = processor(
                audio=audio_ventana, 
                sampling_rate=FRECUENCIA, 
                return_tensors="np" 
            )
            
            # 2. Le pasamos el diccionario de inputs (ya procesado) directamente a TU .onnx
            salidas = sesion_ort.run(None, dict(inputs))
            
            # 3. Imprimimos el tensor resultante
            prediccion = salidas[0]
            print(f"Señal de Namo: {prediccion}")
            
            buffer_audio.clear()

except KeyboardInterrupt:
    print("\nCerrando sistema de escucha.")