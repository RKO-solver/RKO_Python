import multiprocessing
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime

# === Interface Strategy ===
class LogStrategy(ABC):
    @abstractmethod
    def log(self, *args: list[any], **kwargs: dict[str, any]):
        pass

# === Estratégias Concretas ===
class TerminalLogger(LogStrategy):
    """Escreve apenas no terminal."""
    def log(self, *args: list[any], **kwargs: dict[str, any]):
        # flush=True é vital em multiprocessing para não bufferizar a saída
        print(*args, **kwargs, flush=True)

class FileLogger(LogStrategy):
    """Escreve apenas em arquivo."""
    def __init__(self, filepath: str, reset: bool = False):
        self.filepath = filepath
        if reset:
            with open(self.filepath, 'w') as f:
                f.write(f"--- Log Iniciado em {datetime.now()} ---\n")

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        with open(self.filepath, 'a') as f:
            print(*args, **kwargs, file=f)

class DualLogger(LogStrategy):
    """Escreve no Terminal E no Arquivo ao mesmo tempo (Composite Pattern)."""
    def __init__(self, filepath: str, reset: bool = False):
        self.terminal = TerminalLogger()
        self.file = FileLogger(filepath, reset)

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        self.terminal.log(*args, **kwargs)
        self.file.log(*args, **kwargs)

# === Gerenciador de Processos ===
class ParallelLogManager:
    def __init__(self, strategy: LogStrategy):
        self.strategy = strategy
        self.queue = multiprocessing.Manager().Queue()
        self.stop_event = multiprocessing.Event()
        self.listener_process = None

    def start(self):
        self.listener_process = multiprocessing.Process(
            target=self._listener_worker,
            args=(self.queue, self.strategy, self.stop_event)
        )
        self.listener_process.start()

    def stop(self):
        self.stop_event.set()
        if self.listener_process:
            self.listener_process.join()

    def get_logger(self):
        """Retorna o objeto que será passado para os workers."""
        return WorkerLogger(self.queue)

    @staticmethod
    def _listener_worker(queue, strategy, stop_event):
        while not stop_event.is_set() or not queue.empty():
            while not queue.empty():
                try:
                    payload = queue.get()
                    args, kwargs = payload                    
                    strategy.log(*args, **kwargs)
                except Exception as e:
                    print(f"ERRO CRÍTICO NO LOGGER: {e}", file=sys.stderr)
            time.sleep(0.05)

# === Adaptador para os Workers ===
class WorkerLogger:
    def __init__(self, queue):
        self.queue = queue

    def log(self, *args, **kwargs):
        self.queue.put((args, kwargs))
