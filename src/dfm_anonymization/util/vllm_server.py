import subprocess
import time
import threading
import signal
import requests
from openai import OpenAI

class VLLMServer:
    def __init__(self, model_name: str, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.process = None
        self.client = None

        # Additional vLLM args (e.g., gpu_memory_utilization=0.8)
        self.extra_args = kwargs

    def start(self, timeout: int = 300):
        """Starts the vLLM server in a subprocess and waits for it to be ready."""
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port)
        ]

        for key, value in self.extra_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        print(f"🚀 Starting vLLM server: {' '.join(cmd)}")

        # Use start_new_session to ensure we can kill the whole process tree later
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True
        )

        # Wait for the server to signal it's ready
        start_time = time.time()
        success = False

        while time.time() - start_time < timeout:
            line = self.process.stdout.readline()
            if line:
                print(f"[vLLM Log] {line.strip()}")
                if "Uvicorn running on" in line:
                    success = True
                    break
            if self.process.poll() is not None:
                raise RuntimeError("vLLM server failed to start. Check logs above.")

        if not success:
            self.stop()
            raise TimeoutError("vLLM server took too long to start.")

        self.client = OpenAI(base_url=self.base_url, api_key="token-not-needed")
        print(f"✅ Server is ready at {self.base_url}")

    def chat_completion(self, messages, **kwargs):
        """Wraps the OpenAI-compatible chat completion call."""
        if not self.client:
            raise RuntimeError("Server not started. Call .start() first.")

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def stop(self):
        """Gracefully shuts down the server and its worker processes."""
        if self.process and self.process.poll() is None:
            print(f"🛑 Shutting down vLLM server (PID {self.process.pid})...")
            # Kill the entire process group (vLLM spawns many workers)
            import os
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=20)
            print("Cleaned up successfully.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# --- Usage Example ---
if __name__ == "__main__":
    # Context manager handles 'start' and 'stop' automatically
    with VLLMServer("facebook/opt-125m", gpu_memory_utilization=0.5) as server:
        response = server.chat_completion(
            messages=[{"role": "user", "content": "Explain quantum physics in one sentence."}],
            max_tokens=50
        )
        print(f"\nAI Response: {response.choices[0].message.content}")
        pass
