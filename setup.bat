@echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=3.9
set PYTHON_URL=https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe
set GITHUB_REPO=https://raw.githubusercontent.com/phanlequangthao/Khoa-H-c/master
set FILES_TO_DOWNLOAD=model_7.h5,mainlstm.py,client_camera.py,server.py,main.ui,requirements.txt
set FOLDERS_TO_DOWNLOAD=img1,img2
set PROJECT_FOLDER=%~dp0Project
set TEMP_FOLDER=%~dp0Temp

python --version 2>nul || (
    echo "Python %PYTHON_VERSION% chua duoc cai dat, dang tai python..."
    echo Downloading from: %PYTHON_URL%
    curl -L -o python-3.9.0-amd64.exe %PYTHON_URL%
    start /wait python-3.9.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
)
python --version 2>nul || (
    echo "Loi khong the tai duoc python, hay tai va cai dat thu cong thong qua url sau: %PYTHON_URL%"
    exit /b 1
)

if not exist %PROJECT_FOLDER% (
    mkdir %PROJECT_FOLDER%
)

if not exist %TEMP_FOLDER% (
    mkdir %TEMP_FOLDER%
)

for %%f in (%FILES_TO_DOWNLOAD%) do (
    if not exist %PROJECT_FOLDER%\%%f (
        echo Down from: %GITHUB_REPO%/%%f
        curl -L -o %PROJECT_FOLDER%\%%f %GITHUB_REPO%/%%f
    ) else (
        echo File %%f da ton tai, kiem tra commit moi nhat.
        curl -L -o %TEMP_FOLDER%\%%f %GITHUB_REPO%/%%f
        fc %PROJECT_FOLDER%\%%f %TEMP_FOLDER%\%%f >nul
        if errorlevel 1 (
            echo File %%f co su thay doi, cap nhat file.
            copy /y %TEMP_FOLDER%\%%f %PROJECT_FOLDER%\%%f
        ) else (
            echo File %%f khong co thay doi, bo qua.
        )
        del %TEMP_FOLDER%\%%f
    )
)

for %%d in (%FOLDERS_TO_DOWNLOAD%) do (
    if not exist %PROJECT_FOLDER%\%%d (
        mkdir %PROJECT_FOLDER%\%%d
    )
    for %%i in (a b c d e f g h i j k l m n o p q r s t u v w x y z space) do (
        if not exist %PROJECT_FOLDER%\%%d\%%i.jpg (
            echo Downloading from: %GITHUB_REPO%/%%d/%%i.jpg
            curl -L -o %PROJECT_FOLDER%\%%d\%%i.jpg %GITHUB_REPO%/%%d/%%i.jpg
        ) else (
            echo File %%i.jpg trong thu muc %%d da ton tai, bo qua viec setup phan mem.
        )
    )
)

cd %PROJECT_FOLDER%
python -m pip install -r requirements.txt

:run_mainlstm
python %PROJECT_FOLDER%\mainlstm.py