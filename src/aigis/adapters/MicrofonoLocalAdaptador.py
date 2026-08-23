import queue
import sounddevice as sd
import numpy as np

# Este sería tu Adaptador de Micrófono (adaptadores/microfono_local.py)
class MicrofonoLocalAdaptador:
    def __init__(self, frecuencia: int = 16000, chunk_size: int = 512, canales: int = 1):
        self.frecuencia = frecuencia
        self.chunk_size = chunk_size
        self.canales = canales
        # La cola se esconde aquí dentro, el orquestador no sabe que existe
        self.cola_audio = queue.Queue()

    def _callback_microfono(self, indata, frames, time_info, status):
        """Esta función secreta la llama la tarjeta de sonido, no el orquestador."""
        chunk = indata.flatten().astype(np.float32)
        self.cola_audio.put(chunk)

    def escuchar(self):
        """
        Este es el método público que usará el Caso de Uso.
        Es un generador (yield) que escupe audio continuamente.
        """
        print("🎙️ [HARDWARE] Abriendo canal de audio...")
        with sd.InputStream(
            samplerate=self.frecuencia, 
            channels=self.canales, 
            blocksize=self.chunk_size, 
            dtype=np.float32, 
            callback=self._callback_microfono
        ):
            while True:
                # El yield pausa la ejecución hasta que haya un chunk nuevo en la cola
                yield self.cola_audio.get()