import sounddevice as sd
import numpy as np
import onnxruntime as ort
import threading
import queue
import time
from transformers import AutoTokenizer
import onnx_asr

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS LOCALES
# ==========================================
FRECUENCIA = 16000
CHUNK_SIZE = 512
CANALES = 1

RUTA_SILERO = "/home/david/models/silero_vad.onnx"
RUTA_PARAKEET_DIR = "/home/david/models"
RUTA_NAMO = "/home/david/models/Namo-Turn-Detector-v1-Multilingual.onnx"

print("🔥 [INICIALIZANDO AIGIS: PIPELINE NATIVO ONNX] 🔥")

# ==========================================
# 2. CARGA DE MODELOS
# ==========================================
# A. Silero VAD (RAM / CPU)
print("-> Cargando Silero VAD...")
silero_sess = ort.InferenceSession(RUTA_SILERO)
silero_state = np.zeros((2, 1, 128), dtype=np.float32)

# B. Parakeet TDT (STT Multilingüe nativo mediante onnx-asr)
print("-> Cargando Parakeet TDT desde la carpeta local...")
stt_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "RUTA_PARAKEET_DIR", quantization="int8")

# C. Namo Turn Detector (Árbitro semántico)
print("-> Cargando Namo Turn Detector...")
tokenizer = AutoTokenizer.from_pretrained("videosdk-live/Namo-Turn-Detector-v1-Multilingual")
namo_sess = ort.InferenceSession(RUTA_NAMO)

print("¡SISTEMAS ALINEADOS Y OPERATIVOS!\n")

# ==========================================
# 3. ESTADOS Y COLAS
# ==========================================
cola_audio = queue.Queue()
buffer_conversacion = []

estado_voz = False
silence_start_time = None
GRACE_PERIOD = 1  # Segundos de gracia

# ==========================================
# 4. HILO DE CAPTURA DE AUDIO
# ==========================================
def callback_microfono(indata, frames, time_info, status):
    if status:
        pass
    chunk = indata.flatten().astype(np.float32)
    cola_audio.put(chunk)

def hilo_captura():
    with sd.InputStream(device=27, samplerate=FRECUENCIA, channels=CANALES, blocksize=CHUNK_SIZE, dtype=np.float32, callback=callback_microfono):
        while True:
            sd.sleep(100)

thread_micro = threading.Thread(target=hilo_captura, daemon=True)
thread_micro.start()
print("🎙️ Micrófono escuchando en segundo plano...")
print("Habla con naturalidad. El pipeline está completamente listo.\n")

# ==========================================
# 5. BUCLE PRINCIPAL DE LA TUBERÍA
# ==========================================
try:
    while True:
        chunk = cola_audio.get()
        
        # --- PASO A: SILERO VAD ---
        audio_tensor = np.expand_dims(chunk, axis=0)
        entradas_silero = {
            'input': audio_tensor,
            'sr': np.array([FRECUENCIA], dtype=np.int64),
            'state': silero_state
        }
        salidas_silero = silero_sess.run(None, entradas_silero)
        prob_voz = salidas_silero[0][0][0]
        silero_state = salidas_silero[1]
        
        # --- PASO B: BÚFER Y VENTANA DE GRACIA ---
        if prob_voz > 0.60:
            if not estado_voz:
                print("\n[🎙️ Voz detectada...]")
                estado_voz = True
            buffer_conversacion.append(chunk)
            silence_start_time = None
        else:
            if estado_voz:
                buffer_conversacion.append(chunk)
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                elif time.time() - silence_start_time > GRACE_PERIOD:
                    print("[🛑 Fin de turno. Procesando...]")
                    estado_voz = False
                    
                    audio_completo = np.concatenate(buffer_conversacion)
                    buffer_conversacion = []
                    silence_start_time = None
                    
                    # --- PASO C: PARAKEET (STT) ---
                    print("⚡ Transcribiendo con Parakeet...")
                    # onnx-asr acepta arrays numpy float32 directamente
                    texto_generado = stt_model.recognize(audio_completo, sample_rate=FRECUENCIA, language="es")
                    
                    if not texto_generado or not isinstance(texto_generado, str):
                        print("[Audio vacío o ruido descartado]")
                        continue
                        
                    texto_generado = texto_generado.strip()
                    print(f"📝 Texto: \"{texto_generado}\"")
                    
                    # --- PASO D: NAMO TURN DETECTOR ---
                    print("🧠 Consultando al Turn Detector...")
                    inputs_namo = tokenizer(
                        texto_generado, 
                        return_tensors="np", 
                        padding=True, 
                        truncation=True
                    )
                    
                    salidas_namo = namo_sess.run(None, {
                        'input_ids': inputs_namo['input_ids'],
                        'attention_mask': inputs_namo['attention_mask']
                    })
                    
                    prediccion_namo = salidas_namo[0]
                    print(f"⚖️ Veredicto de Namo: {prediccion_namo}")
                    print("-" * 50)

except KeyboardInterrupt:
    print("\n🛑 Apagando Aigis con éxito.")