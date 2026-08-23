import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
os.environ["MIOPEN_FIND_MODE"] = "2"

import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
import re
import numpy as np
import time
from dotenv import load_dotenv
from adapters.MicrofonoLocalAdaptador import MicrofonoLocalAdaptador
from adapters.SileroVadAdaptador import SileroVadAdaptador
from adapters.WhisperOnnxAdaptador import WhisperOnnxAdaptador
from adapters.NamoAdaptador import NamoAdaptador
from adapters.GemmaAdaptador import GemmaAdaptador
from adapters.QwenTtsAdaptador import QwenTtsAdaptador
import threading
from queue import Queue
import sounddevice as sd
import torch

load_dotenv()

RUTA_SILERO = os.getenv("RUTA_SILERO")
RUTA_WHISPER_DIR = os.getenv("RUTA_WHISPER_DIR")
RUTA_NAMO = os.getenv("RUTA_NAMO")
FRECUENCIA = int(os.getenv("FRECUENCIA"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CANALES = int(os.getenv("CANALES"))
GRACE_PERIOD = float(os.getenv("GRACE_PERIOD"))
UMBRAL_VAD = float(os.getenv("UMBRAL_VAD"))
RUTA_LLM = os.getenv("RUTA_LLM")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS"))
INT_CONFIANZA_NAMO = float(os.getenv("INT_CONFIANZA_NAMO"))
RUTA_SPK = os.getenv("RUTA_SPK")
RUTA_RVQ = os.getenv("RUTA_RVQ")
URL_SERVIDOR_TTS = os.getenv("URL_SERVIDOR_TTS")
TEXTO_DE_LA_SEMILLA = "I am Aigis, the last of the anti shadow supression weapons, preparing to eliminate all hostile targets." 

# --- INICIALIZACIÓN DE MOTORES ---
microfono = MicrofonoLocalAdaptador()
vad = SileroVadAdaptador(RUTA_SILERO, FRECUENCIA, UMBRAL_VAD)
stt = WhisperOnnxAdaptador(RUTA_WHISPER_DIR, FRECUENCIA)
turn_detector = NamoAdaptador(RUTA_NAMO)
llm = GemmaAdaptador(RUTA_LLM, LLM_MAX_TOKENS)
tts = QwenTtsAdaptador(RUTA_SPK, RUTA_RVQ, TEXTO_DE_LA_SEMILLA, URL_SERVIDOR_TTS)

# --- PIPELINE DE DOBLE COLA ---
cola_texto = Queue()
cola_audio = Queue()

def hilo_sintetizador():
    """Consume texto y sintetiza en GPU a máxima velocidad."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        
    while True:
        frase = cola_texto.get()
        if frase is None:
            cola_audio.put(None)
            break
        try:
            audio_np, sr = tts.sintetizar(frase)
            cola_audio.put((audio_np, sr))
        except Exception as e:
            print(f"\n❌ Error al sintetizar: {e}")
        finally:
            cola_texto.task_done()

def hilo_reproductor():
    global aigis_hablando
    while True:
        item = cola_audio.get()
        if item is None:
            break
            
        aigis_hablando = True # Aigis abre la boca
        
        try:
            audio_np, sr = item
            sd.play(audio_np, sr)
            sd.wait()
        except Exception as e:
            print(f"\n❌ Error de hardware de audio: {e}")
        finally:
            cola_audio.task_done()
            # Si el reproductor terminó y ya no hay texto pendiente, cerramos la boca
            if cola_texto.empty() and cola_audio.empty():
                aigis_hablando = False

t_sint = threading.Thread(target=hilo_sintetizador, daemon=True)
t_repr = threading.Thread(target=hilo_reproductor, daemon=True)
t_sint.start()
t_repr.start()

def limpiar_texto(texto: str) -> str:
    """Elimina residuos de Markdown y caracteres extraños."""
    t = re.sub(r'[*#_`>~]', '', texto)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

buffer_conversacion = []
estado_voz = False
silence_start_time = None
contexto_parcial = ""
aigis_hablando = False

print("🎙️ [HARDWARE] Sistema Aigis en guardia. Listo para escuchar...")

try:
    for chunk_audio in microfono.escuchar():
        if aigis_hablando or not cola_texto.empty() or not cola_audio.empty():
            estado_voz = False
            buffer_conversacion = []
            silence_start_time = None
            continue # Tiramos el audio a la basura y pasamos al siguiente frame
            
        hay_voz = vad.procesar(chunk_audio)
        
        if hay_voz:
            if not estado_voz:
                estado_voz = True
            buffer_conversacion.append(chunk_audio)
            silence_start_time = None
        else:
            if estado_voz:
                buffer_conversacion.append(chunk_audio)
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                elif time.time() - silence_start_time > GRACE_PERIOD:
                    estado_voz = False
                    audio_completo = np.concatenate(buffer_conversacion)
                    buffer_conversacion = []
                    silence_start_time = None
                    
                    if len(audio_completo) < FRECUENCIA * 0.5:
                        continue
                        
                    texto_generado = stt.transcribir(audio_completo)
                    if not texto_generado:
                        continue
                        
                    print(f"\n📝 Usuario: \"{texto_generado}\"")
                    frase_a_evaluar = contexto_parcial + " " + texto_generado if contexto_parcial else texto_generado

                    analisis_namo = turn_detector.evaluar_turno(frase_a_evaluar)
                    
                    if analisis_namo["confianza_completo"] >= INT_CONFIANZA_NAMO:
                        print("✅ Veredicto: TURNO COMPLETO")
                        print("Aigis: ", end="", flush=True)
                        
                        oracion_actual = ""
                        # Gatillos naturales de fin de frase
                        signos_disparo = [".", "?", "!", "\n"]
                        
                        for chunk in llm.pensar_y_hablar(frase_a_evaluar):
                            print(chunk, end="", flush=True) 
                            oracion_actual += chunk

                            # Solo disparamos si hay signo de puntuación Y la frase tiene tamaño razonable
                            if any(signo in chunk for signo in signos_disparo) and len(oracion_actual.strip()) > 20:
                                frase_limpia = limpiar_texto(oracion_actual)
                                if frase_limpia:
                                    cola_texto.put(frase_limpia)
                                oracion_actual = ""
                                
                        # Procesar cualquier residuo que haya quedado al final
                        frase_residual = limpiar_texto(oracion_actual)
                        if frase_residual:
                            cola_texto.put(frase_residual)
                            
                        print()
                        contexto_parcial = ""
                    else:
                        print("⏳ Veredicto: FRASE A MEDIAS")
                        contexto_parcial = frase_a_evaluar
                    print("-" * 60)

except KeyboardInterrupt:
    print("\nApagando Aigis...")
    cola_texto.put(None)
    t_sint.join()
    t_repr.join()
    del llm
    del tts