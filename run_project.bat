@echo off
call .venv\Scripts\activate
python -m src.pipeline --data data\PJME_hourly.csv --value-col PJME_MW --epochs 80 --window 24 --stride 6 --output results
pause
