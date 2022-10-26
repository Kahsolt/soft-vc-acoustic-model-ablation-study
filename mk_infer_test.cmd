@REM 2022/09/14 
@REM Generate all
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET PYTHON_BIN=python


%PYTHON_BIN% infer.py baseline_databaker-full     test

%PYTHON_BIN% infer.py baseline_databaker-8h       test
%PYTHON_BIN% infer.py baseline_databaker-4h       test
%PYTHON_BIN% infer.py baseline_databaker-2h       test
%PYTHON_BIN% infer.py baseline_databaker-1h       test
%PYTHON_BIN% infer.py baseline_databaker-30min    test
%PYTHON_BIN% infer.py baseline_databaker-10min    test

%PYTHON_BIN% infer.py no_dropout_databaker-full   test 
%PYTHON_BIN% infer.py no_IN_databaker-full        test
%PYTHON_BIN% infer.py single_LSTM_databaker-full  test
%PYTHON_BIN% infer.py only_Encoder_databaker-full test
%PYTHON_BIN% infer.py only_Decoder_databaker-full test
%PYTHON_BIN% infer.py tiny_databaker-full         test
%PYTHON_BIN% infer.py tiny_half_databaker-full    test
