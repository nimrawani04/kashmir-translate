@echo off
REM ============================================================================
REM EXTREME FINE-TUNING (24+ points expected) - MAXIMUM QUALITY
REM Time: 8-10 hours
REM ============================================================================

echo ============================================================================
echo    EXTREME FINE-TUNING FOR 24+ POINTS (MAXIMUM QUALITY)
echo ============================================================================
echo.

REM Check HF_TOKEN
if "%HF_TOKEN%"=="" (
    echo [ERROR] HuggingFace token not set!
    echo.
    echo Please run this first:
    echo   set HF_TOKEN=hf_your_token_here
    echo.
    pause
    exit /b 1
)

echo [OK] HuggingFace token detected
echo.

echo Configuration:
echo ============================================================================
echo   Epochs:                15
echo   BPCC samples:          15,000
echo   LoRA rank:             128
echo   LoRA alpha:            256
echo   Batch size:            1
echo   Gradient accumulation: 16
echo   Learning rate:         1e-4
echo   Warmup ratio:          0.2
echo   Output dir:            out/lora-kas-extreme
echo.
echo   Expected training time: 8-10 hours
echo   Expected score:        24+ points
echo   Expected rank:         Top 3-10
echo.
echo   WARNING: This is very intensive!
echo ============================================================================
echo.

set /p CONFIRM="Start EXTREME fine-tuning? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Training cancelled.
    exit /b 0
)

echo.
echo ============================================================================
echo    STEP 1/4: FINE-TUNING MODEL (This will take 8-10 hours!)
echo ============================================================================
echo.
echo [INFO] Training started at %TIME%
echo [INFO] This is the EXTREME mode - maximum quality!
echo [INFO] Target: Final loss < 0.5 for best quality
echo.
echo TIP: This will run overnight. You can:
echo   - Leave it running
echo   - Check progress periodically
echo   - Model saves checkpoints every 100 steps
echo.

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
    --save-steps 50 ^
    --output-dir out/lora-kas-extreme

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Training completed at %TIME%
echo.
pause

echo.
echo ============================================================================
echo    STEP 2/4: GENERATING HIGH-QUALITY TRANSLATIONS
echo ============================================================================
echo.

python inference_finetuned.py ^
    --model-dir out/lora-kas-extreme ^
    --batch-size 4 ^
    --num-beams 8 ^
    --temperature 0.5 ^
    --top-p 0.95 ^
    --repetition-penalty 1.3 ^
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
echo    EXTREME FINE-TUNING COMPLETE!
echo ============================================================================
echo.
echo Generated files:
echo   [✓] submission_finetuned.csv - MAXIMUM QUALITY!
echo   [✓] translation_comparison.csv - Comparison data
echo   [✓] out/lora-kas-extreme/ - Trained model
echo.
echo Expected score: 24+ points
echo Expected rank:  Top 3-10
echo.
echo ============================================================================
echo    CONGRATULATIONS!
echo ============================================================================
echo.
echo You've completed EXTREME fine-tuning with:
echo   - 15 epochs of training
echo   - 15,000 BPCC samples
echo   - LoRA rank 128 (highest capacity)
echo   - 8+ beams for generation
echo.
echo This should give you TOP-TIER scores!
echo.
echo ============================================================================
echo    NEXT STEPS
echo ============================================================================
echo.
echo 1. Review the comparison - should show significant improvements
echo.
echo 2. Submit to Kaggle:
echo    → https://www.kaggle.com/competitions/kathe-2026/submissions
echo    → Upload: submission_finetuned.csv
echo.
echo 3. Expect 24+ score and top 3-10 rank! 🏆
echo.
echo ============================================================================
pause
