@echo off
REM ============================================================================
REM Fine-tuning Commands for KATHE 2026 - Maximum Score (20+)
REM ============================================================================

echo ============================================================================
echo    KATHE 2026 - Fine-tuning Commands for 20+ Score
echo ============================================================================
echo.

REM Check HF_TOKEN
if "%HF_TOKEN%"=="" (
    echo [ERROR] HuggingFace token not set!
    echo.
    echo Set it with:
    echo   set HF_TOKEN=hf_your_token_here
    echo.
    echo Get token from: https://huggingface.co/settings/tokens
    echo.
    pause
    exit /b 1
)

echo [OK] HuggingFace token detected
echo.

REM ============================================================================
echo CHOOSE YOUR FINE-TUNING STRATEGY:
echo ============================================================================
echo.
echo [1] STANDARD (3-4 hours, 20-22 points)
echo     - 5 epochs, 5K BPCC samples
echo     - Good balance of speed and quality
echo.
echo [2] AGGRESSIVE (5-6 hours, 22-24 points) - RECOMMENDED
echo     - 10 epochs, 10K BPCC samples, higher LoRA rank
echo     - Maximum quality
echo.
echo [3] EXTREME (8-10 hours, 24+ points)
echo     - 15 epochs, 15K BPCC samples, highest LoRA rank
echo     - Absolute maximum quality
echo.
echo [4] QUICK TEST (30 min, test run)
echo     - 1 epoch, 1K BPCC samples
echo     - For testing setup
echo.

set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" goto STANDARD
if "%CHOICE%"=="2" goto AGGRESSIVE
if "%CHOICE%"=="3" goto EXTREME
if "%CHOICE%"=="4" goto QUICK
echo Invalid choice!
pause
exit /b 1

REM ============================================================================
:STANDARD
REM ============================================================================
echo.
echo ============================================================================
echo    STANDARD FINE-TUNING (20-22 points expected)
echo ============================================================================
echo.
echo Configuration:
echo   - Epochs: 5
echo   - BPCC samples: 5,000
echo   - LoRA rank: 32
echo   - Batch size: 2
echo   - Gradient accumulation: 8
echo   - Learning rate: 3e-4
echo   - Training time: ~3-4 hours
echo.

set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" exit /b 0

python finetune_improved.py ^
    --epochs 5 ^
    --use-bpcc ^
    --bpcc-samples 5000 ^
    --batch-size 2 ^
    --grad-accum 8 ^
    --lora-r 32 ^
    --lora-alpha 64 ^
    --lr 3e-4 ^
    --warmup-ratio 0.1 ^
    --output-dir out/lora-kas-standard

goto INFERENCE

REM ============================================================================
:AGGRESSIVE
REM ============================================================================
echo.
echo ============================================================================
echo    AGGRESSIVE FINE-TUNING (22-24 points expected) - RECOMMENDED
echo ============================================================================
echo.
echo Configuration:
echo   - Epochs: 10
echo   - BPCC samples: 10,000
echo   - LoRA rank: 64
echo   - Batch size: 2
echo   - Gradient accumulation: 8
echo   - Learning rate: 2e-4
echo   - Training time: ~5-6 hours
echo.

set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" exit /b 0

python finetune_improved.py ^
    --epochs 10 ^
    --use-bpcc ^
    --bpcc-samples 10000 ^
    --batch-size 2 ^
    --grad-accum 8 ^
    --lora-r 64 ^
    --lora-alpha 128 ^
    --lr 2e-4 ^
    --warmup-ratio 0.15 ^
    --output-dir out/lora-kas-aggressive

goto INFERENCE

REM ============================================================================
:EXTREME
REM ============================================================================
echo.
echo ============================================================================
echo    EXTREME FINE-TUNING (24+ points expected)
echo ============================================================================
echo.
echo Configuration:
echo   - Epochs: 15
echo   - BPCC samples: 15,000
echo   - LoRA rank: 128
echo   - Batch size: 1 (for stability)
echo   - Gradient accumulation: 16
echo   - Learning rate: 1e-4
echo   - Training time: ~8-10 hours
echo.
echo WARNING: This is very intensive and may overfit!
echo.

set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" exit /b 0

python finetune_improved.py ^
    --epochs 15 ^
    --use-bpcc ^
    --bpcc-samples 15000 ^
    --batch-size 1 ^
    --grad-accum 16 ^
    --lora-r 128 ^
    --lora-alpha 256 ^
    --lr 1e-4 ^
    --warmup-ratio 0.2 ^
    --output-dir out/lora-kas-extreme

goto INFERENCE

REM ============================================================================
:QUICK
REM ============================================================================
echo.
echo ============================================================================
echo    QUICK TEST (Testing setup only)
echo ============================================================================
echo.
echo Configuration:
echo   - Epochs: 1
echo   - BPCC samples: 1,000
echo   - LoRA rank: 16
echo   - Training time: ~30 minutes
echo.

set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" exit /b 0

python finetune_improved.py ^
    --epochs 1 ^
    --use-bpcc ^
    --bpcc-samples 1000 ^
    --batch-size 2 ^
    --grad-accum 4 ^
    --lora-r 16 ^
    --lora-alpha 32 ^
    --lr 3e-4 ^
    --output-dir out/lora-kas-quick

goto INFERENCE

REM ============================================================================
:INFERENCE
REM ============================================================================

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo    Training Complete! Now generating translations...
echo ============================================================================
echo.

REM Determine output directory based on choice
set MODEL_DIR=out/lora-kas-standard
if "%CHOICE%"=="2" set MODEL_DIR=out/lora-kas-aggressive
if "%CHOICE%"=="3" set MODEL_DIR=out/lora-kas-extreme
if "%CHOICE%"=="4" set MODEL_DIR=out/lora-kas-quick

echo Using model from: %MODEL_DIR%
echo.

python inference_finetuned.py ^
    --model-dir %MODEL_DIR% ^
    --batch-size 4 ^
    --num-beams 5 ^
    --temperature 0.6 ^
    --top-p 0.9 ^
    --repetition-penalty 1.2 ^
    --output submission_finetuned.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Inference failed!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo    Validating submission...
echo ============================================================================
echo.

python validate_submission.py

echo.
echo ============================================================================
echo    Comparing with manual translations...
echo ============================================================================
echo.

python compare_translations.py

echo.
echo ============================================================================
echo    SUCCESS! Fine-tuning pipeline complete!
echo ============================================================================
echo.
echo Generated files:
echo   - submission_finetuned.csv (main output)
echo   - translation_comparison.csv (comparison data)
echo.
echo Next steps:
echo   1. Review the comparison results
echo   2. Upload submission_finetuned.csv to Kaggle
echo   3. Wait for score (expected 20-24+ points!)
echo.
echo Competition URL: https://www.kaggle.com/competitions/kathe-2026
echo.
pause
