from datetime import datetime

class SimpleFileLogger:
    def __init__(self, filename="logs/app_log.txt", mode='a'):
        """
        Initialize a simple file logger
        
        Args:
            filename: Name of the log file
            mode: File mode ('a' for append, 'w' for overwrite)
        """
        self.filename = filename
        self.mode = mode
        # Write initialization message
        self._write(f"\n\n=== Log session started at {self._timestamp()} ===\n")
    
    def _timestamp(self):
        """Generate current timestamp"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _write(self, message):
        """Internal write method"""
        with open(self.filename, self.mode, encoding='utf-8') as f:
            f.write(message)
    
    def log(self, message, level="INFO"):
        """Log a message with specified level"""
        log_entry = f"[{self._timestamp()}] [{level}] {message}\n"
        self._write(log_entry)
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def debug(self, message):
        self.log(message, "DEBUG")