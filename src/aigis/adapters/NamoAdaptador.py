import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class NamoAdaptador:
    def __init__(self, ruta_modelo: str):
        print("⚙️ [ADAPTADOR] Iniciando Namo Turn Detector...")
        # El tokenizador siempre se descarga/cachea desde Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained("videosdk-live/Namo-Turn-Detector-v1-Multilingual")
        # El modelo pesado cargado en RAM desde tu disco duro
        self.session = ort.InferenceSession(ruta_modelo)

    def evaluar_turno(self, texto: str) -> bool:
        """
        Recibe un texto y devuelve True si el usuario ha terminado de hablar,
        o False si se ha quedado a medias.
        """
        inputs = self.tokenizer(
            texto, 
            return_tensors="np", 
            padding=True, 
            truncation=True
        )
        
        salidas = self.session.run(None, {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })
        
        logits = salidas[0][0] 
        
        exp_logits = np.exp(logits - np.max(logits)) # Restamos el máximo por estabilidad
        probabilidades = exp_logits / exp_logits.sum()
        
        prob_incompleto = probabilidades[0] * 100
        prob_completo = probabilidades[1] * 100
        
        # Empaquetamos todo el conocimiento en un diccionario
        return {
            "es_completo": prob_completo > prob_incompleto,
            "confianza_completo": prob_completo,
            "confianza_incompleto": prob_incompleto
        }