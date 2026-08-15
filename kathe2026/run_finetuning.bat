@echo off
REM ============================================================================
REM KATHE 2026 - Automated Fine-tuning Pipeline for 20+ Score
REM ============================================================================

echo ============================================================================
echo    KATHE 2026 - Fine-tuning Pipeline for 20+ Score
echo ============================================================================
echo.

REM Check if HF_TOKEN is set
if "%HF_TOKEN%"=="" (
    echo [ERROR] HuggingFace token not set!
    echo.
    echo Please set your HuggingFace token:
    echo    set HF_TOKEN=your_token_here
    echo.
    echo Get token from: https://huggingface.co/settings/tokens
    pause
    exit /b 1
)

echo [INFO] HuggingFace token: %HF_TOKEN:~0,10%...
echo.

REM Step 1: Fine-tune the model
echo ============================================================================
echo    STEP 1: Fine-tuning Model (This will take 3-4 hours)
echo ============================================================================
echo.
echo Configuration:
echo    - Epochs: 5
echo    - Use BPCC: Yes
echo    - BPCC samples: 5000
echo    - Batch size: 2
echo    - Gradient accumulation: 8
echo    - LoRA rank: 32
echo.

set /p CONTINUE="Start fine-tuning? (Y/N): "
if /i not "%CONTINUE%"=="Y" (
    echo Fine-tuning cancelled.
    exit /b 0
)

echo.
echo [INFO] Starting fine-tuning...
echo.

python finetune_improved.py ^
    --epochs 5 ^
    --use-bpcc ^
    --bpcc-samples 5000 ^
    --batch-size 2 ^
    --grad-accum 8 ^
    --lora-r 32 ^
    --lora-alpha 64 ^
    --lr 3e-4 ^
    --output-dir out/lora-kas-improved

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Fine-tuning failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Fine-tuning completed!
echo.
pause

REM Step 2: Generate translations
echo ============================================================================
echo    STEP 2: Generating Translations with Fine-tuned Model
echo ============================================================================
echo.

python inference_finetuned.py ^
    --model-dir out/lora-kas-improved ^
    --batch-size 4 ^
    --num-beams 5 ^
    --temperature 0.6 ^
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

REM Step 3: Validate submissions
echo ============================================================================
echo    STEP 3: Validating Submission
echo ============================================================================
echo.

python validate_submission.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Validation found issues!
    echo Please review the output above.
    echo.
)

echo.
pause

REM Step 4: Compare translations
echo ============================================================================
echo    STEP 4: Comparing Manual vs Fine-tuned Translations
echo ============================================================================
echo.

python compare_translations.py

echo.
pause

REM Final summary
echo.
echo ============================================================================
echo    PIPELINE COMPLETE!
echo ============================================================================
echo.
echo Generated files:
echo    - submission.csv (manual translations)
echo    - submission_finetuned.csv (fine-tuned model translations)
echo    - translation_comparison.csv (comparison analysis)
echo.
echo Next steps:
echo    1. Review comparison results
echo    2. Choose best submission (manual or fine-tuned)
echo    3. Upload to Kaggle: https://www.kaggle.com/competitions/kathe-2026
echo    4. Wait for score (target: 20+ points)
echo.
echo Expected scores:
echo    - Manual submission: 15-18 points
echo    - Fine-tuned submission: 20-24 points
echo.
echo Good luck! ^_^
echo.
pause
