import hashlib
import os
import subprocess
import sys
import time


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ensure_backend_deps():
    """Runs pip install only if requirements.txt has changed since the
    last successful install (tracked via a hash marker file), so repeat
    runs start instantly instead of re-installing every time."""
    req_path = os.path.join("backend", "requirements.txt")
    marker_path = os.path.join("backend", ".deps_installed")

    current_hash = _file_hash(req_path)
    previous_hash = None
    if os.path.exists(marker_path):
        with open(marker_path) as f:
            previous_hash = f.read().strip()

    if current_hash == previous_hash:
        print("[1/4] Backend dependencies already installed and up to date -- skipping.")
        return

    print("[1/4] Installing backend dependencies (pip install -r requirements.txt)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd="./backend")
    if result.returncode != 0:
        print("WARNING: pip install had errors -- check the output above. Continuing anyway.")
        return  # don't write the marker if install failed, so it retries next time

    with open(marker_path, "w") as f:
        f.write(current_hash)


def ensure_frontend_deps():
    """Runs npm install only if node_modules doesn't exist yet or
    package.json has changed since the last install."""
    node_modules_path = os.path.join("frontend", "node_modules")
    pkg_path = os.path.join("frontend", "package.json")
    marker_path = os.path.join("frontend", ".deps_installed")

    current_hash = _file_hash(pkg_path)
    previous_hash = None
    if os.path.exists(marker_path):
        with open(marker_path) as f:
            previous_hash = f.read().strip()

    if os.path.isdir(node_modules_path) and current_hash == previous_hash:
        print("[2/4] Frontend dependencies already installed and up to date -- skipping.")
        return

    print("[2/4] Installing frontend dependencies (npm install)...")
    result = subprocess.run("npm install", cwd="./frontend", shell=True)
    if result.returncode != 0:
        print("WARNING: npm install had errors -- check the output above. Continuing anyway.")
        return

    with open(marker_path, "w") as f:
        f.write(current_hash)


def main():
    print("========================================")
    print("Starting ESMA Dental Analysis System...")
    print("========================================")

    backend_process = None
    frontend_process = None

    try:
        ensure_backend_deps()
        ensure_frontend_deps()

        # We use sys.executable (python -m uvicorn) to avoid
        # a "not recognized" error on Windows
        backend_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload"]

        # Command for the React frontend
        frontend_cmd = "npm run dev"

        print("[3/4] Starting Python FastAPI Backend on port 8000...")
        backend_process = subprocess.Popen(backend_cmd, cwd="./backend")

        # Give it a short delay so the server has time to start properly
        time.sleep(2)

        print("[4/4] Starting Vite React Frontend on port 5173...")
        frontend_process = subprocess.Popen(frontend_cmd, cwd="./frontend", shell=True)

        # Let the script keep running
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n========================================")
        print("Shutting down servers cleanly...")
        print("========================================")
        if backend_process is not None:
            backend_process.terminate()
        if frontend_process is not None:
            frontend_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()