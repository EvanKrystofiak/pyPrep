@echo off
rem Launch the pyPrep desktop app using the project environment
start "" "%~dp0env\pythonw.exe" -m pyprep.gui.launcher
