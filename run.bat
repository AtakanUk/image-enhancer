@echo off
title Image Enhancer
cd /d "%~dp0"
py -3.12 main.py
if errorlevel 1 (
  echo.
  echo The application exited with an error. See the message above.
  pause
)
