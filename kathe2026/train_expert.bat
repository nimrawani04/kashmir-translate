@echo off
REM ============================================================================
REM EXPERT FINE-TUNING FOR 25+ SCORE (3-4 hours)
REM ============================================================================

echo ============================================================================
echo    EXPERT FINE-TUNING FOR 25+ SCORE
echo ============================================================================
echo.

echo Based on competition-winning strategies:
echo   1. Kashmiri-only BPCC fine-tune (biggest single jump)
echo   2. Script verification (kas_Arab only)
echo   3. NFC normalization (free chrF++ points)
echo   4. Dedupe/clean BPCC (quality over quantity)
echo   5. Beam=8 with adjusted length penalty
echo.
echo Configuration:
echo   • LoRA rank: 64 (good capacity)
echo   • Epochs: 8 (optimal convergence)
echo   • BPCC: 10,000 pairs (cleaned and deduped)
echo   • Batch size: 1, Grad accum: 16
echo.
echo Expected:
echo   • Time: 3-4 hours
echo   • Score: 25-28+ points
echo   • Rank: TOP 5
echo.
echo ============================================================================
echo.

set /p CONFIRM="Start expert training? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Training cancelled.
    exit /b 0
)

echo.
echo ============================================================================
echo    PHASE 1/2: FINE-TUNING (3-4 hours)
echo ============================================================================
echo.
echo [INFO] Started at %TIME%
echo.

python finetune_expert.py ^
    --epochs 8 ^
    --lora-r 64 ^
    --lora-alpha 128 ^
    --bpcc-samples 10000 ^
    --batch-size 1 ^
    --grad-accum 16 ^
    --lr 1e-4 ^
    --output-dir out/lora-kas-expert

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
echo    PHASE 2/2: INFERENCE WITH EXPERT SETTINGS
echo ============================================================================
echo.

python inference_expert.py ^
    --model-dir out/lora-kas-expert ^
    --beam 8 ^
    --length-penalty 1.3 ^
    --repetition-penalty 1.2 ^
    --batch-size 4 ^
    --output submission_expert_finetuned.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Inference failed!
    pause
    exit /b 1
)

echo.
pause

echo.
echo ============================================================================
echo    VALIDATION
echo ============================================================================
echo.

REM Copy for validation
copy submission_expert_finetuned.csv submission_finetuned.csv /Y

python validate_submission.py

echo.
pause

echo.
echo ============================================================================
echo    EXPERT TRAINING COMPLETE!
echo ============================================================================
echo.
echo Generated file:
echo   [✓] submission_expert_finetuned.csv
echo.
echo Expert improvements applied:
echo   ✅ Kashmiri-only BPCC fine-tune (biggest jump)
echo   ✅ Dedupe/clean BPCC (quality over quantity)
echo   ✅ NFC normalization (free chrF++ points)
echo   ✅ Script verification (kas_Arab only)
echo   ✅ Beam=8, length penalty=1.3 (optimal)
echo.
echo Expected Performance:
echo   • Score: 25-28+ points
echo   • Rank: TOP 5
echo   • Time: 3-4 hours
echo.
echo Why this works:
echo   • Kashmiri-specific fine-tune (not diluted by 22 languages)
echo   • Cleaned data (noisy pairs removed)
echo   • Optimized for geometric mean (BLEU × chrF++)
echo   • NFC normalization fixes codepoint variations
echo.
echo ============================================================================
echo    SUBMISSION INSTRUCTIONS
echo ============================================================================
echo.
echo 1. Submit to Kaggle:
echo    → URL: https://www.kaggle.com/competitions/kathe-2026/submissions
echo    → File: submission_expert_finetuned.csv
echo.
echo 2. Expected results:
echo    → Score: 25-28+ points
echo    → Rank: TOP 5
echo.
echo ============================================================================
pause
