@echo off
chcp 65001 >nul
echo ============================================
echo Uploading project to GitHub
echo ============================================
echo.

echo Checking GitHub connection...
git remote -v
echo.

echo Uploading code to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo Success! Project uploaded to GitHub
    echo ============================================
    echo.
    echo Repository: https://github.com/Mistifikot/video-prompt-pipeline
) else (
    echo.
    echo ============================================
    echo Error during upload!
    echo ============================================
    echo.
    echo Make sure that:
    echo 1. Repository is created on GitHub: https://github.com/Mistifikot/video-prompt-pipeline
    echo 2. You have write access to the repository
    echo 3. Correct access token is being used
)

pause

