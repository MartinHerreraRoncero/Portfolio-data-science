"""
Lanzador del microservicio RESTful FastAPI sobre servidor ASGI Uvicorn.
Permite iniciar la API directamente ejecutando:
    python Entregable/api/run_api.py
"""

import sys
from pathlib import Path
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
ENTREGABLE_DIR = CURRENT_DIR.parent

# Inserción de directorios prioritarios en sys.path
for path_entry in [str(ENTREGABLE_DIR), str(CURRENT_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

if __name__ == "__main__":
    print("=" * 75)
    print("  Corporate Bankruptcy Prediction API v2.4.0 (S&P 500 / SEC EDGAR)")
    print("  Marcos Regulatorios: Fed SR 11-7 | Basel III | IFRS 9 | EU AI Act")
    print("=" * 75)
    print("  -> Servidor local activo en: http://127.0.0.1:8000")
    print("  -> Cockpit Dashboard GUI:    http://127.0.0.1:8000/ui")
    print("  -> Documentacion Swagger UI: http://127.0.0.1:8000/docs")
    print("  -> Documentacion ReDoc:      http://127.0.0.1:8000/redoc")
    print("=" * 75)

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=str(CURRENT_DIR)
    )
