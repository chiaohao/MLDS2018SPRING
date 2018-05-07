wget https://www.dropbox.com/s/kkw1fjp967udx49/word_dict_best.txt 
wget https://www.dropbox.com/s/4c6dj43zp9h97gb/decoder_best.pt 
wget https://www.dropbox.com/s/e8opfuvwkdaqpvb/encoder_best.pt 
python3 test_2gru_attn.py word_dict_best.txt encoder_best.pt decoder_best.pt $1 $2 0 0

