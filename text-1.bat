@echo off

setlocal enabledelayedexpansion
set ARGS=
set POS=cmd
for %%a in (%*) do (
  if "!POS!"=="cmd" (
    set COMMAND=%%~a
    set POS=arg
  ) else if "!POS!"=="arg" (
    set ARGS=!ARGS! %%a
  )
)

python application.py %COMMAND% ^
--embeddings-pickle-file text1-embeddings.pkl ^
--index-mask-pickle-file text1-index-mask.pkl ^
--cat-target-key Positive ^
--create-args-cat-csv-file yes_data.csv ^
--dog-target-key Negative ^
--create-args-dog-csv-file no_data.csv ^
--model-id amazon.titan-embed-text-v1 ^
--embedding-vector-dimensions 1536 %ARGS%

endlocal
