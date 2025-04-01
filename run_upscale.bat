@echo off
cd /d %~dp0
call env\Scripts\activate.bat
python scripts\video_sr_webui.py
pause
