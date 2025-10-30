@echo off
echo ============================================
echo Загрузка проекта на GitHub
echo ============================================
echo.

echo Проверка подключения к GitHub...
git remote -v
echo.

echo Загрузка кода на GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo Успешно! Проект загружен на GitHub
    echo ============================================
    echo.
    echo Репозиторий: https://github.com/Mistifikot/video-prompt-pipeline
) else (
    echo.
    echo ============================================
    echo Ошибка при загрузке!
    echo ============================================
    echo.
    echo Убедитесь что:
    echo 1. Репозиторий создан на GitHub: https://github.com/Mistifikot/video-prompt-pipeline
    echo 2. У вас есть права на запись в репозиторий
    echo 3. Используется правильный токен доступа
)

pause

