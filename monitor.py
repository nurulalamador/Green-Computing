import time
import psutil
import platform
import subprocess
import threading


def get_cpu_temperature():
    system = platform.system()

    if system == "Linux":
        try:
            output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
            return float(output.replace("temp=", "").replace("'C", "").strip())
        except Exception:
            pass

        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as file:
                temp = float(file.read()) / 1000.0
                return temp
        except Exception:
            return None

    return None


class SystemMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.records = []
        self.running = False
        self.thread = None

    def _monitor(self):
        psutil.cpu_percent(interval=None)

        while self.running:
            record = {
                "time": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "temperature_c": get_cpu_temperature()
            }

            self.records.append(record)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join()

    def summary(self):
        if not self.records:
            return {
                "avg_cpu_percent": 0,
                "max_cpu_percent": 0,
                "avg_memory_percent": 0,
                "max_memory_percent": 0,
                "avg_temperature_c": None,
                "max_temperature_c": None
            }

        cpu_values = [r["cpu_percent"] for r in self.records]
        memory_values = [r["memory_percent"] for r in self.records]
        temperature_values = [
            r["temperature_c"] for r in self.records
            if r["temperature_c"] is not None
        ]

        return {
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_percent": sum(memory_values) / len(memory_values),
            "max_memory_percent": max(memory_values),
            "avg_temperature_c": (
                sum(temperature_values) / len(temperature_values)
                if temperature_values else None
            ),
            "max_temperature_c": (
                max(temperature_values)
                if temperature_values else None
            )
        }