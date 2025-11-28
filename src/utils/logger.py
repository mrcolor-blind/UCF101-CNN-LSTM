# Impresión bonita / logs
# src/utils/logger.py

import os
from datetime import datetime

class Logger:
    """
    Logger sencillo para imprimir mensajes en consola
    y opcionalmente guardarlos en un archivo.
    """

    def __init__(self, log_file=None):
        self.log_file = log_file

        # Crear directorio automáticamente si log_file ya viene desde config
        if log_file:
            folder = os.path.dirname(log_file)
            if folder != "" and not os.path.exists(folder):
                os.makedirs(folder)

    # ========= HELPERS =========

    def _write_to_file(self, msg):
        if self.log_file is None:
            return

        # Asegurar que la carpeta existe incluso si log_file fue asignado después
        folder = os.path.dirname(self.log_file)
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format(self, level, message):
        return f"[{self._timestamp()}] [{level}] {message}"

    # ========= PUBLIC METHODS =========

    def info(self, message):
        msg = self._format("INFO", message)
        print(msg)
        self._write_to_file(msg)

    def warn(self, message):
        msg = self._format("WARN", message)
        print(msg)
        self._write_to_file(msg)

    def error(self, message):
        msg = self._format("ERROR", message)
        print(msg)
        self._write_to_file(msg)

    def success(self, message):
        msg = self._format("SUCCESS", message)
        print(msg)
        self._write_to_file(msg)


# Logger global
logger = Logger()
