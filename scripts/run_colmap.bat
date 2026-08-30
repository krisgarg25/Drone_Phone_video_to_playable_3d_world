@echo off
REM Piece 2: COLMAP sequential reconstruction from keyframes.
REM Thin delegate - the implementation lives in run_colmap.py.
REM Usage: run_colmap.bat <work_dir>
setlocal
set HERE=%~dp0
"%HERE%..\.venv\Scripts\python.exe" "%HERE%run_colmap.py" %*
exit /b %errorlevel%
