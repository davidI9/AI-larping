import numpy as np
from transformers import AutoProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

class WhisperOnnxAdaptador:
    def __init__(self, ruta_modelo: str, frecuencia: int = 16000):
        print("⚙️ [ADAPTADOR] Cargando Whisper Turbo ONNX...")
        self.frecuencia = frecuencia
        self.processor = AutoProcessor.from_pretrained(ruta_modelo)
        self.modelo = ORTModelForSpeechSeq2Seq.from_pretrained(
            ruta_modelo, 
            provider="CPUExecutionProvider",
            use_merged=False
        )

    def transcribir(self, audio: np.ndarray) -> str:
        """Convierte el audio crudo en texto puro y blindado."""
        inputs = self.processor(
            audio, 
            sampling_rate=self.frecuencia, 
            return_tensors="pt",
            return_attention_mask=True
        )
        
        generated_ids = self.modelo.generate(
            inputs["input_features"], 
            attention_mask=inputs["attention_mask"],
            language="es"
        )
        
        texto = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Filtro de diccionario explícito (opcional pero recomendado)
        return texto.replace("Aegis", "Aigis").replace("Iris", "Aigis")