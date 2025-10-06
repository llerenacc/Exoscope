# =====================================
# run_app.ps1 — Activa venv y corre Streamlit
# =====================================

# Ruta a tu entorno virtual
$venvPath = "C:/Users/Rosa Campana/Documents/Exoscope/venv/Scripts/Activate.ps1"

# Activar entorno virtual
Write-Host "🔹 Activando entorno virtual..."
& $venvPath

# Espera un segundo para asegurar activación
Start-Sleep -Seconds 1

# Ejecutar Streamlit
Write-Host "🔹 Corriendo Streamlit..."
streamlit run "C:/Users/Rosa Campana/Documents/Exoscope/app/Phase 6.py"
