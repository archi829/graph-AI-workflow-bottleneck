# steps
- create n activate venv
- pip install each requirement separately (doing via rqmnts.txt hangs due to dependency resolution)
- download ollama.exe -> install ollama -> check version
    > run in command prompt to keep ollama server running 
    set OLLAMA_HOST=0.0.0.0:11434
    ollama serve
- >commands to run:
    ollama pull phi3:mini
    python run_batch.py --system crewai --n 3