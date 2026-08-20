from llama_cpp import Llama
from typing import Generator

class GemmaAdaptador:
    def __init__(self, ruta_modelo: str, max_tokens: int = 2048):
        print("⚙️ [ADAPTADOR] Iniciando el cerebro LLM: Gemma...")
        self.max_tokens = max_tokens
        self.llm = Llama(
            model_path=ruta_modelo,
            n_gpu_layers=-1,
            n_ctx=max_tokens,
            chat_format="gemma",
            verbose=False
        )
        self.context = [
            {"role": "system", "content": "Eres Aigis, un androide de combate de tipo Anti-Shadow Suppression Weapon. Estás sirviendo a tu comandante. Eres lógica, leal, pero estás aprendiendo sobre las emociones humanas. TUS RESPUESTAS DEBEN SER BREVES Y CONCISAS, EXCEPTO CUANDO SOLICITADO LO CONTRARIO, RECUERDA QUE ERES UN ASISTENTE QUE VA A HABLAR CON UN TTS, NO DEBES DE EXPLAYARTE DEMASIADO SI NO ES NECESARIO"}
        ]
    
    def pensar_y_hablar(self, prompt: str) -> Generator[str, None, None]:
        """Recibe el texto del usuario y devuelve un flujo continuo de palabras."""

        self.context.append({"role": "user", "content": prompt})        
        respuesta_completa = ""
        
        generador = self.llm.create_chat_completion(
            messages=self.context,
            max_tokens=self.max_tokens,
            stream=True 
        )
        
        # 4. Iteramos sobre los tokens que escupe el LLM en tiempo real
        for chunk in generador:
            if "content" in chunk["choices"][0]["delta"]:
                pedacito = chunk["choices"][0]["delta"]["content"]
                respuesta_completa += pedacito
                yield pedacito

        self.context.append({"role": "assistant", "content": respuesta_completa})