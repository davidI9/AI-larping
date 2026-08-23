import sounddevice as sd
import numpy as np
import requests
import base64
import io
import wave

class QwenTtsAdaptador:
    def __init__(self, ruta_spk: str, ruta_rvq: str, texto_semilla: str, url_servidor: str = "http://localhost:8080"):
        print("⚙️ [ADAPTADOR] Iniciando conexión con Microservicio C++ (GGML)...")
        self.url = url_servidor
        self.voice_name = "aigis_voice"
        
        # 1. Leemos los latentes pre-calculados que forjaste en la Fase 3
        with open(ruta_spk, "rb") as f:
            spk_b64 = base64.b64encode(f.read()).decode('utf-8')
        with open(ruta_rvq, "rb") as f:
            rvq_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        # 2. Registramos la voz de Aigis en el servidor C++ (Ocurre en milisegundos)
        print("🧬 [VOZ] Registrando latentes GGML en el servidor...")
        payload = {
            "name": self.voice_name,
            "ref_text": texto_semilla,
            "spk_b64": spk_b64,
            "rvq_b64": rvq_b64
        }
        resp = requests.post(f"{self.url}/v1/audio/voices", json=payload)
        if resp.status_code == 200:
            print("✅ [VOZ] Aigis registrada y lista en el servidor C++.")
        else:
            print(f"❌ Error al registrar voz: {resp.text}")

    def sintetizar(self, texto: str) -> tuple[np.ndarray, int]:
        """Envía el texto al servidor C++ y recibe el audio compilado al instante."""
        payload = {
            "input": texto,
            "voice": self.voice_name,
            "response_format": "wav",
            "temperature": 0.7 # Ajusta para que suene más robótica o más natural
        }
        
        # Hacemos la petición al motor de C++
        resp = requests.post(f"{self.url}/v1/audio/speech", json=payload)
        
        if resp.status_code == 200:
            # Extraemos el audio puro del WAV que nos devuelve el servidor
            with io.BytesIO(resp.content) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    sr = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    # qwentts.cpp devuelve PCM 16-bit
                    audio_np = np.frombuffer(frames, dtype=np.int16) 
            return audio_np, sr
        else:
            print(f"❌ Error de síntesis: {resp.text}")
            return np.array([]), 24000