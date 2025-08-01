@echo off
REM Activate virtual environment and set Quarto Python path
call .venv\Scripts\activate
set QUARTO_PYTHON=.venv\Scripts\python.exe

REM Run Quarto preview with the specified file
quarto preview conference_talks/2025_MC/template.qmd --no-browser --no-watch-inputs

REM Keep the window open if there's an error
pause 