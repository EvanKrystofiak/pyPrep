@echo off
rem Launch the pyPrep desktop app using the project environment (created by install.bat)
if not exist "%~dp0env\pythonw.exe" (
  echo pyPrep is not installed yet - run install.bat in this folder first.
  pause
  exit /b 1
)
start "" "%~dp0env\pythonw.exe" -m pyprep.gui.launcher
