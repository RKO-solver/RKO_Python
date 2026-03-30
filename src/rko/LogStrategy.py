import multiprocessing
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime

# === Interface Strategy ===
class LogStrategy(ABC):
    """
    Abstract Base Class for logging strategies.
    
    Defines the standard interface for any logging strategy used in the framework.
    """

    @abstractmethod
    def log(self, *args: list[any], **kwargs: dict[str, any]):
        """
        Logs the provided arguments.

        Args:
            *args (list[any]): Variable length argument list to be logged.
            **kwargs (dict[str, any]): Arbitrary keyword arguments (e.g., end='\\n').
        """
        pass

# === Estratégias Concretas ===
class TerminalLogger(LogStrategy):
    """
    Logging strategy that writes output exclusively to the terminal.
    
    Useful for interactive monitoring of the optimization process.
    """

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        """
        Prints the provided arguments to the standard output.
        
        Args:
            *args (list[any]): Variable length argument list to be printed.
            **kwargs (dict[str, any]): Arbitrary keyword arguments.
        """
        # flush=True is vital in multiprocessing to prevent output buffering
        print(*args, **kwargs, flush=True)

class FileLogger(LogStrategy):
    """
    Logging strategy that writes output exclusively to a specified file.
    
    Useful for keeping persistent records of the optimization runs.
    """

    def __init__(self, filepath: str, reset: bool = False):
        """
        Initializes the file logger.

        Args:
            filepath (str): The path to the file where logs should be written.
            reset (bool, optional): If True, overwrites the existing file. 
                If False, appends to the existing file. Defaults to False.
        """
        self.filepath = filepath
        if reset:
            with open(self.filepath, 'w') as f:
                f.write(f"--- Log Iniciado em {datetime.now()} ---\n")

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        """
        Appends the provided arguments to the configured log file.

        Args:
            *args (list[any]): Variable length argument list to be written.
            **kwargs (dict[str, any]): Arbitrary keyword arguments.
        """
        with open(self.filepath, 'a') as f:
            print(*args, **kwargs, file=f)

class DualLogger(LogStrategy):
    """
    Composite logging strategy that writes to both the terminal and a file simultaneously.
    """

    def __init__(self, filepath: str, reset: bool = False):
        """
        Initializes the dual logger.

        Args:
            filepath (str): The path to the file where logs should be written.
            reset (bool, optional): If True, overwrites the existing log file. 
                Defaults to False.
        """
        self.terminal = TerminalLogger()
        self.file = FileLogger(filepath, reset)

    def log(self, *args: list[any], **kwargs: dict[str, any]):
        """
        Logs the provided arguments to both the terminal and the file.

        Args:
            *args (list[any]): Variable length argument list to be logged.
            **kwargs (dict[str, any]): Arbitrary keyword arguments.
        """
        self.terminal.log(*args, **kwargs)
        self.file.log(*args, **kwargs)

# === Gerenciador de Processos ===
class ParallelLogManager:
    """
    Manages logging asynchronously in a multiprocessing environment.
    
    Creates a dedicated listener process that receives log messages from 
    different workers via a thread-safe Queue, preventing race conditions.
    """

    def __init__(self, strategy: LogStrategy):
        """
        Initializes the parallel log manager.

        Args:
            strategy (LogStrategy): The concrete logging strategy to be used 
                (e.g., TerminalLogger, FileLogger, DualLogger).
        """
        self.strategy = strategy
        self.queue = multiprocessing.Manager().Queue()
        self.stop_event = multiprocessing.Event()
        self.listener_process = None

    def start(self):
        """
        Starts the dedicated listener daemon process for handling logs.
        """
        self.listener_process = multiprocessing.Process(
            target=self._listener_worker,
            args=(self.queue, self.strategy, self.stop_event)
        )
        self.listener_process.start()

    def stop(self):
        """
        Signals the listener process to terminate and waits for it to finish gracefully.
        """
        self.stop_event.set()
        if self.listener_process:
            self.listener_process.join()

    def get_logger(self):
        """
        Produces a lightweight proxy logger instance to be passed to parallel workers.

        Returns:
            WorkerLogger: A logger proxy that sends messages to the central queue.
        """
        return WorkerLogger(self.queue)

    @staticmethod
    def _listener_worker(queue, strategy, stop_event):
        """
        Background worker method that consumes log messages from the queue.

        Args:
            queue (multiprocessing.Queue): The queue holding incoming log payloads.
            strategy (LogStrategy): The strategy responsible for executing the logging action.
            stop_event (multiprocessing.Event): Event to signal termination.
        """
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
    """
    Proxy logger passed to worker processes to offload log actions to the main queue.
    """

    def __init__(self, queue):
        """
        Initializes the worker logger.

        Args:
            queue (multiprocessing.Queue): The shared queue connected to the LogManager.
        """
        self.queue = queue

    def log(self, *args, **kwargs):
        """
        Puts the log payload onto the queue for execution by the central listener.

        Args:
            *args: Variable length argument list to be logged.
            **kwargs: Arbitrary keyword arguments.
        """
        self.queue.put((args, kwargs))
