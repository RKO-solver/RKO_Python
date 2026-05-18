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
    """Escreve no terminal E opcionalmente em logs.txt."""
    def __init__(self, logs_filepath: str | None = None, reset: bool = False):
        self.logs_filepath = logs_filepath
        if self.logs_filepath and reset:
            with open(self.logs_filepath, 'w', encoding='utf-8') as f:
                f.write(f"--- Log Started at {datetime.now().strftime('%d/%m/%Y %H:%M')} ---\n")

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        msg = " ".join(str(arg) for arg in args)
        
        if msg.startswith("[info]"):
            # Print para o terminal (onde o best_pair[0] já vem em verde)
            print(msg, **kwargs, flush=True)
            return
            
        if msg.startswith("[best]"):
            try:
                parts = msg.split(" | ")
                metaheuristic_name = parts[0].replace("[best] ", "").strip()
                fitness = parts[1].strip()
                elapsed_time = parts[2].strip()
                pool_len = parts[3].strip()
                
                # Terminal: print beautiful green [best] message
                term_msg = f"\033[32m[best] {metaheuristic_name} find a solution with fitness {fitness}, is the new best solution! time: {elapsed_time}s\033[0m"
                print(term_msg, **kwargs, flush=True)
                
                # File: write clean old-school format (without colors)
                if self.logs_filepath:
                    file_msg = f"{metaheuristic_name} NEW BEST: {fitness} - Time: {elapsed_time}s - {pool_len}"
                    with open(self.logs_filepath, 'a', encoding='utf-8') as f:
                        print(file_msg, file=f, flush=True)
            except Exception:
                # Fallback em caso de falha no parse
                print(msg, **kwargs, flush=True)
                if self.logs_filepath:
                    with open(self.logs_filepath, 'a', encoding='utf-8') as f:
                        print(msg, file=f, flush=True)
            return
            
        # Qualquer outro tipo de log (cabeçalhos, rodapés, etc.)
        print(msg, **kwargs, flush=True)
        if self.logs_filepath:
            with open(self.logs_filepath, 'a', encoding='utf-8') as f:
                print(msg, file=f, flush=True)

class FileLogger(LogStrategy):
    """Escreve apenas em arquivo."""
    def __init__(self, filepath: str, reset: bool = False):
        self.filepath = filepath
        if reset:
            with open(self.filepath, 'w') as f:
                f.write(f"--- Log Started at {datetime.now().strftime('%d/%m/%Y %H:%M')} ---\n")

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
