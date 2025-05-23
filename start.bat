@echo off
echo Starting DIY AutoML Platform...

echo Starting backend server...
start cmd /k "cd research paper && cd diyml && cd backend && myenv\Scripts\activate && pip install -r requirements.txt && python app.py"

echo Starting frontend...
start cmd /k "cd research paper && cd diyml && cd frontend && npm install && npm run dev"

echo DIY AutoML Platform is starting up.
echo Backend will be available at http://localhost:5000
echo Frontend will be available at http://localhost:3000