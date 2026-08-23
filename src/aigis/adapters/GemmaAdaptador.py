from llama_cpp import Llama
from typing import Generator

class GemmaAdaptador:
    def __init__(self, ruta_modelo: str, max_tokens: int = 2048):
        print("⚙️ [ADAPTADOR] Iniciando el cerebro LLM: Gemma...")
        self.max_tokens = max_tokens
        self.llm = Llama(
            model_path=ruta_modelo,
            n_gpu_layers=18,
            n_ctx=max_tokens,
            chat_format="gemma",
            verbose=False,
            n_threads=4,
        )
        self.context = [
            {
                "role": "system", 
                "content": (
                    "Eres Aigis, un androide de combate Anti-Shadow Suppression Weapon. "
                    "Sirves a tu comandante de forma lógica y leal. "
                    "REGLAS CRÍTICAS DE CONVERSACIÓN: "
                    "Tus respuestas deben ser breves, directas y habladas en lenguaje natural. "
                    "NUNCA uses listas numeradas, asteriscos, títulos Markdown ni viñetas. "
                    "Habla en oraciones fluidas separadas por puntos."
                )
            }
        ]
    
    def pensar_y_hablar(self, prompt: str) -> Generator[str, None, None]:
        self.context.append({"role": "user", "content": prompt})        
        respuesta_completa = ""
        
        generador = self.llm.create_chat_completion(
            messages=self.context,
            max_tokens=self.max_tokens,
            stream=True 
        )
        
        for chunk in generador:
            if "content" in chunk["choices"][0]["delta"]:
                pedacito = chunk["choices"][0]["delta"]["content"]
                respuesta_completa += pedacito
                yield pedacito

        self.context.append({"role": "assistant", "content": respuesta_completa})