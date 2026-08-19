import re
from llama_cpp import Llama

# 1. Inicializamos el LLM
print("Cargando el cerebro de Aigis en la VRAM... (RX 9060XT trabajando)")
llm = Llama(
    model_path="/home/david/models/gemma-4-12b-it-Q5_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=8000, # Aumentamos el contexto para que recuerde más conversación
    chat_format="gemma",
    verbose=False
)
print("¡Carga completada!\n")

# 2. Definimos la personalidad inicial
historial_mensajes = [
    {"role": "system", "content": "Eres Aigis, un androide de combate de tipo Anti-Shadow Suppression Weapon. Estás sirviendo a tu comandante. Eres lógica, leal, pero estás aprendiendo sobre las emociones humanas. Tus respuestas deben ser directas y sin florituras innecesarias."}
]

# Abrimos el archivo de log (opcional, por si quieres mantener el registro)
log_file = open("historial_conversacion.md", "a", encoding="utf-8")

# 3. EL BUCLE PRINCIPAL
while True:
    try:
        # Esperamos tu entrada por terminal (más adelante esto será el VAD+STT)
        mensaje_usuario = input("\n[Tú]: ")
        
        # Comando para salir del bucle
        if mensaje_usuario.lower() in ["salir", "exit", "apagar"]:
            print("Apagando sistemas...")
            break
            
        # Añadimos tu mensaje al historial
        historial_mensajes.append({"role": "user", "content": mensaje_usuario})
        log_file.write(f"\n\n**Comandante:** {mensaje_usuario}")
        
        print("[Aigis]: ", end="", flush=True)
        log_file.write(f"\n\n**Aigis:** ")
        
        # Generamos la respuesta con streaming
        flujo = llm.create_chat_completion(
            messages=historial_mensajes,
            max_tokens=8000, # Puedes ajustarlo según necesites
            stream=True 
        )

        respuesta_completa = ""
        es_pensamiento = False

        for chunk in flujo:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                texto_nuevo = delta["content"]
                
                # Filtro de pensamientos ocultos
                if "<|channel>thought" in texto_nuevo:
                    es_pensamiento = True
                    continue 
                if "<channel|>" in texto_nuevo or "</channel>" in texto_nuevo:
                    es_pensamiento = False
                    continue 
                if es_pensamiento:
                    continue 

                # Limpieza
                texto_limpio = re.sub(r'[*_~"\[\]()]', '', texto_nuevo) 
                texto_limpio = re.sub(r'<[^>]+>', '', texto_limpio)
                
                respuesta_completa += texto_limpio
                print(texto_limpio, end="", flush=True) # Mostrar en terminal
                
                log_file.write(texto_limpio)
                log_file.flush()
        
        # Añadimos la respuesta final de Aigis al historial para que tenga contexto en la siguiente ronda
        historial_mensajes.append({"role": "assistant", "content": respuesta_completa})
        
    except KeyboardInterrupt:
        # Por si pulsas Ctrl+C
        print("\n\nInterrupción forzada. Apagando sistemas...")
        break

log_file.close()