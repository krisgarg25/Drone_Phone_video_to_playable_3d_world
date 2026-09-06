@echo off
REM ============================================================
REM  Drone video -> 3D Gaussian Splat -> walkable scene
REM  Thin shim over pipeline.py - the single source of step order.
REM  Usage:  mvp.bat                      one command: the dashboard (run/view/capture)
REM          mvp.bat run <name> [flags]   e.g. mvp.bat run rocks
REM          mvp.bat status <name>
REM          mvp.bat view <name>          serve + open the viewer
REM          mvp.bat doctor               toolchain + both interpreters + node voxeliser
REM          mvp.bat check                every fast suite (budgets, capture, collider, gate)
REM          mvp.bat check e2e            + every take in videos\ at --quality smoke
REM  Flags:  --quality high|standard|smoke (default high; smoke = every step, minutes)
REM          --preset auto|room|drone|...  (default auto: it diagnoses your footage)
REM          --fresh | --from <step> | --only <step,step>
REM          --timeout-scale F            slower hardware: 1.5; shakedown: 0.5
REM          --steps N --cap N --width N --target N   overrides
REM ============================================================
setlocal
set ROOT=%~dp0
set PY=%ROOT%.venv\Scripts\python.exe

if "%~1"=="" goto :ui
if /i "%~1"=="view" goto :view
if /i "%~1"=="check" goto :check

"%PY%" "%ROOT%pipeline.py" %*
exit /b %errorlevel%

:ui
"%PY%" "%ROOT%pipeline.py" ui
exit /b %errorlevel%

:check
shift
if /i "%~1"=="e2e" (
  "%PY%" "%ROOT%tests\check_all.py" --e2e %2 %3 %4
) else (
  "%PY%" "%ROOT%tests\check_all.py" %1 %2 %3 %4
)
exit /b %errorlevel%

:view
if "%~2"=="" goto :usage
"%PY%" "%ROOT%pipeline.py" view %~2
exit /b %errorlevel%

:usage
echo usage: mvp.bat                     (open the dashboard)
echo        mvp.bat run ^<video_name^> [--quality standard] [--from STEP]
echo        mvp.bat status ^<video_name^>
echo        mvp.bat view ^<video_name^>
exit /b 1
