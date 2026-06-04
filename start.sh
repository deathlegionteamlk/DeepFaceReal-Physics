#!/bin/bash

CREDITS="Powered By DeathLegionTeamLK"

echo "╔══════════════════════════════════════════════╗"
echo "║     DeepFaceReal Physics - Startup Script    ║"
echo "║     $CREDITS          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Start FastAPI backend on port 8081
echo "[START] Launching FastAPI backend on port 8081..."
cd "$(dirname "$0")"
python3 api.py &
API_PID=$!
echo "[START] API PID: $API_PID"

# Start Streamlit dashboard on port 8080
echo "[START] Launching Streamlit dashboard on port 8080..."
streamlit run app.py --server.port=8080 --server.address=0.0.0.0 &
STREAMLIT_PID=$!
echo "[START] Streamlit PID: $STREAMLIT_PID"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     Services Running                        ║"
echo "║     Streamlit: http://localhost:8080         ║"
echo "║     FastAPI:   http://localhost:8081         ║"
echo "║     API Docs:  http://localhost:8081/docs    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Trap to clean up on exit
trap "kill $API_PID $STREAMLIT_PID 2>/dev/null; echo 'Shutdown complete'" EXIT

# Wait for either process to exit
wait