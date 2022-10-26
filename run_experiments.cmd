python train.py databaker --listfp lists\databaker-full.txt

python train.py databaker --listfp lists\databaker-8h.txt
python train.py databaker --listfp lists\databaker-4h.txt
python train.py databaker --listfp lists\databaker-2h.txt
python train.py databaker --listfp lists\databaker-1h.txt
python train.py databaker --listfp lists\databaker-30min.txt
python train.py databaker --listfp lists\databaker-10min.txt

python train.py databaker --listfp lists\databaker-full.txt --model no_dropout
python train.py databaker --listfp lists\databaker-full.txt --model no_IN
python train.py databaker --listfp lists\databaker-full.txt --model single_LSTM
python train.py databaker --listfp lists\databaker-full.txt --model only_Encoder
python train.py databaker --listfp lists\databaker-full.txt --model only_Decoder
python train.py databaker --listfp lists\databaker-full.txt --model tiny
python train.py databaker --listfp lists\databaker-full.txt --model tiny_half
