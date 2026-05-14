@echo off
REM ============================================
REM PrimoAgent 批量回测脚本
REM 3只股票 × 40天 → 自动回测
REM ============================================
setlocal

set VENV=venv\Scripts\python.exe
set START=2026-03-02
set END=2026-04-30

echo ================================================
echo  PrimoAgent 批量回测
echo  股票: AAPL MSFT GOOGL
echo  周期: %START% ~ %END%
echo  开始时间: %date% %time%
echo ================================================

for %%S in (AAPL MSFT GOOGL) do (
    echo.
    echo ========================================
    echo  正在分析: %%S
    echo ========================================
    set PRIMO_SYMBOL=%%S
    set PRIMO_START_DATE=%START%
    set PRIMO_END_DATE=%END%
    %VENV% -u main.py
    if errorlevel 1 (
        echo [FAIL] %%S 分析失败
    ) else (
        echo [OK] %%S 分析完成
    )
)

echo.
echo ================================================
echo  所有股票分析完成，开始回测...
echo ================================================

REM 非交互式多股票回测
(echo 2 && echo 1 && echo n && echo n) | %VENV% backtest.py

echo.
echo ================================================
echo  全部完成! %date% %time%
echo ================================================
endlocal
