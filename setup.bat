@echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=3.9
set PYTHON_URL=https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe
set GITHUB_REPO=https://raw.githubusercontent.com/phanlequangthao/nhatKHKT/refs/heads/main/
set FILES_TO_DOWNLOAD=best_model_12.h5,mainlstm.py,client_camera.py,server.py,main.ui,requirements.txt
set FOLDERS_TO_DOWNLOAD=img1,img2
set PROJECT_FOLDER=%~dp0Project
set TEMP_FOLDER=%~dp0Temp

python --version 2>nul || (
    echo "Python %PYTHON_VERSION% chưa được cài đặt, đang tải Python..."
    curl -L -o python-3.9.0-amd64.exe %PYTHON_URL%
    start /wait python-3.9.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
)

python --version 2>nul || (
    echo "Lỗi không thể tải được Python - hãy tải và cài đặt thủ công qua URL sau: %PYTHON_URL%"
    exit /b 1
)

if not exist %PROJECT_FOLDER% mkdir %PROJECT_FOLDER%
if not exist %TEMP_FOLDER% mkdir %TEMP_FOLDER%
for %%f in (%FILES_TO_DOWNLOAD%) do (
    echo Checking file: %%f
    curl -L -o %TEMP_FOLDER%\%%f %GITHUB_REPO%/%%f
    if not exist %PROJECT_FOLDER%\%%f (
        echo Downloading new file: %%f
        move /y %TEMP_FOLDER%\%%f %PROJECT_FOLDER%\%%f
    ) else (
        fc %PROJECT_FOLDER%\%%f %TEMP_FOLDER%\%%f >nul
        if errorlevel 1 (
            echo Updating file: %%f
            move /y %TEMP_FOLDER%\%%f %PROJECT_FOLDER%\%%f
        ) else (
            echo File %%f không có thay đổi - skip.
            del %TEMP_FOLDER%\%%f
        )
    )
)
for %%d in (%FOLDERS_TO_DOWNLOAD%) do (
    if not exist %PROJECT_FOLDER%\%%d mkdir %PROJECT_FOLDER%\%%d
    for %%i in (a b c d e f g h i j k l m n o p q r s t u v w x y z space) do (
        if not exist %PROJECT_FOLDER%\%%d\%%i.jpg (
            echo Downloading from: %GITHUB_REPO%/%%d/%%i.jpg
            curl -L -o %PROJECT_FOLDER%\%%d\%%i.jpg %GITHUB_REPO%/%%d/%%i.jpg
        ) else (
            echo File %%i.jpg trong thư mục %%d đã tồn tại, bỏ qua.
        )
    )
)
cd %PROJECT_FOLDER%
python -m pip install -r requirements.txt
:run_mainlstm
python %PROJECT_FOLDER%\mainlstm.py