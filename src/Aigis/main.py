import numpy as np
import time
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import os
from dotenv import load_dotenv
from adapters.MicrofonoLocalAdaptador import MicrofonoLocalAdaptador
from adapters.SileroVadAdaptador import SileroVadAdaptador
from adapters.WhisperOnnxAdaptador import WhisperOnnxAdaptador
from adapters.NamoAdaptador import NamoAdaptador
from adapters.GemmaAdaptador import GemmaAdaptador

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

buffer_conversacion = []
estado_voz = False
silence_start_time = None
contexto_parcial = ""

microfono = MicrofonoLocalAdaptador()
vad = SileroVadAdaptador(RUTA_SILERO, FRECUENCIA, UMBRAL_VAD)
stt = WhisperOnnxAdaptador(RUTA_WHISPER_DIR, FRECUENCIA)
turn_detector = NamoAdaptador(RUTA_NAMO)
llm = GemmaAdaptador(RUTA_LLM, LLM_MAX_TOKENS)

try:
    for chunk_audio in microfono.escuchar():
        
        hay_voz=vad.procesar(chunk_audio)
        
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
                        
                    print(f"📝 Texto: \"{texto_generado}\"")
                    
                    frase_a_evaluar = contexto_parcial + " " + texto_generado if contexto_parcial else texto_generado

                    analisis_namo = turn_detector.evaluar_turno(frase_a_evaluar)
                    es_turno_completo = analisis_namo["es_completo"]
                    prob_comp = analisis_namo["confianza_completo"]
                    prob_inc = analisis_namo["confianza_incompleto"]
                    
                    if prob_comp >= INT_CONFIANZA_NAMO:
                        print("✅ Veredicto: TURNO COMPLETO (Aigis debería responder)")
                        print("Aigis: ", end="", flush=True)
                        for chunk in llm.pensar_y_hablar(frase_a_evaluar):
                            print(chunk, end="", flush=True)
                        print()
                        contexto_parcial = ""
                    else:
                        print("⏳ Veredicto: FRASE A MEDIAS (Aigis debería seguir escuchando)")
                        contexto_parcial = frase_a_evaluar
                    print("-" * 60)

except KeyboardInterrupt:
    print("\nApagando Aigis.") 

