@echo off
REM ============================================================
REM  Drone video -> 3D Gaussian Splat -> walkable scene
REM  Thin shim over pipeline.py - the single source of step order.
REM  Usage:  mvp.bat run <name> [flags]   e.g. mvp.bat run rocks
REM          mvp.bat status <name>
REM          mvp.bat view <name>          serve + open the viewer
REM  Flags:  --quality high|standard  (default high)
REM          --fresh | --from <step> | --only <step,step>
REM          --steps N --cap N --width N --target N   overrides
REM ============================================================
setlocal
set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if "%~1"=="" goto :usage
if /i "%~1"=="view" goto :view

"%PY%" "%ROOT%pipeline.py" %*
exit /b %errorlevel%

:view
if "%~2"=="" goto :usage
"%PY%" "%ROOT%pipeline.py" view %~2
exit /b %errorlevel%

:usage
echo usage: mvp.bat run ^<video_name^> [--quality standard] [--from STEP]
echo        mvp.bat status ^<video_name^>
echo        mvp.bat view ^<video_name^>
exit /b 1
