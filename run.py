import subprocess
import sys
import time

def main():
    print("========================================")
    print("Starting ESMA Dental Analysis System...")
    print("========================================")
    
    # Gumagamit tayo ng sys.executable (python -m uvicorn) 
    # para maiwasan yung "not recognized" na error sa Windows
    backend_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload"]
    
    # Command para sa React
    frontend_cmd = "npm run dev"
    
    try:
        print("[1/2] Starting Python FastAPI Backend on port 8000...")
        backend_process = subprocess.Popen(backend_cmd, cwd="./backend")
        
        # Bigyan ng konting delay para maka-start nang maayos ang server
        time.sleep(2) 
        
        print("[2/2] Starting Vite React Frontend on port 5173...")
        frontend_process = subprocess.Popen(frontend_cmd, cwd="./frontend", shell=True)
        
        # Hayaan lang tumakbo ang script
        backend_process.wait()
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\n========================================")
        print("Shutting down servers cleanly...")
        print("========================================")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()