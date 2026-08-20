import numpy as np
import onnxruntime as ort

class SileroVadAdaptador:
    def __init__(self, ruta_modelo: str, frecuencia: int = 16000, umbral: float = 0.75):
        print("⚙️ [ADAPTADOR] Iniciando Silero VAD...")
        self.frecuencia = frecuencia
        self.umbral = umbral
        
        # El modelo y SU MEMORIA quedan ocultos aquí dentro
        self.session = ort.InferenceSession(ruta_modelo)
        self.estado_interno = np.zeros((2, 1, 128), dtype=np.float32)

    def procesar(self, chunk_audio: np.ndarray) -> bool:
        """Recibe audio puro y devuelve True (Voz) o False (Silencio)"""
        audio_tensor = np.expand_dims(chunk_audio, axis=0)
        
        entradas = {
            'input': audio_tensor,
            'sr': np.array([self.frecuencia], dtype=np.int64),
            'state': self.estado_interno
        }
        
        salidas = self.session.run(None, entradas)
        
        # Actualizamos la memoria interna silenciosamente
        prob_voz = salidas[0][0][0]
        self.estado_interno = salidas[1]
        
        return prob_voz > self.umbral