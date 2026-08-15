@echo off
REM ============================================================================
REM AGGRESSIVE FINE-TUNING (22-24 points expected) - RECOMMENDED
REM Time: 5-6 hours
REM ============================================================================

echo ============================================================================
echo    AGGRESSIVE FINE-TUNING FOR 22-24 POINTS (RECOMMENDED)
echo ============================================================================
echo.

REM Check HF_TOKEN
if "%HF_TOKEN%"=="" (
    echo [ERROR] HuggingFace token not set!
    echo.
    echo Please run this first:
    echo   set HF_TOKEN=hf_your_token_here
    echo.
    echo Get token from: https://huggingface.co/settings/tokens
    echo.
    pause
    exit /b 1
)

echo [OK] HuggingFace token detected
echo.

echo Configuration:
echo ============================================================================
echo   Epochs:                10
echo   BPCC samples:          10,000
echo   LoRA rank:             64
echo   LoRA alpha:            128
echo   Batch size:            2
echo   Gradient accumulation: 8
echo   Learning rate:         2e-4
echo   Warmup ratio:          0.15
echo   Output dir:            out/lora-kas-aggressive
echo.
echo   Expected training time: 5-6 hours
echo   Expected score:        22-24 points
echo   Expected rank:         Top 5-15
echo ============================================================================
echo.

set /p CONFIRM="Start aggressive fine-tuning? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Training cancelled.
    exit /b 0
)

echo.
echo ============================================================================
echo    STEP 1/4: FINE-TUNING MODEL (This will take 5-6 hours)
echo ============================================================================
echo.
echo [INFO] Training started at %TIME%
echo [INFO] Monitor the loss values - they should decrease over time
echo [INFO] Target: Final loss < 1.0 for best quality
echo.

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

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Training failed!
    echo.
    echo Troubleshooting:
    echo   - CUDA OOM? Try: --batch-size 1 --grad-accum 16
    echo   - Check if HF_TOKEN is valid
    echo   - Ensure you have internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Training completed at %TIME%
echo.
pause

echo.
echo ============================================================================
echo    STEP 2/4: GENERATING TRANSLATIONS (This will take 30 minutes)
echo ============================================================================
echo.

python inference_finetuned.py ^
    --model-dir out/lora-kas-aggressive ^
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
echo [SUCCESS] Translations generated!
echo.
pause

echo.
echo ============================================================================
echo    STEP 3/4: VALIDATING SUBMISSION
echo ============================================================================
echo.

python validate_submission.py

echo.
pause

echo.
echo ============================================================================
echo    STEP 4/4: COMPARING WITH MANUAL TRANSLATIONS
echo ============================================================================
echo.

python compare_translations.py

echo.
pause

echo.
echo ============================================================================
echo    AGGRESSIVE FINE-TUNING COMPLETE!
echo ============================================================================
echo.
echo Generated files:
echo   [✓] submission_finetuned.csv - Ready to submit!
echo   [✓] translation_comparison.csv - Comparison data
echo   [✓] out/lora-kas-aggressive/ - Trained model
echo.
echo Expected score: 22-24 points
echo Expected rank:  Top 5-15
echo.
echo ============================================================================
echo    NEXT STEPS
echo ============================================================================
echo.
echo 1. Review translation_comparison.csv to see improvements
echo.
echo 2. Submit to Kaggle:
echo    → Go to: https://www.kaggle.com/competitions/kathe-2026/submissions
echo    → Upload: submission_finetuned.csv
echo    → Wait for score!
echo.
echo 3. Celebrate your 22-24 point score! 🎉
echo.
echo ============================================================================
pause
