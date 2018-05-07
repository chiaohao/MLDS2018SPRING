CUDA_VISIBLE_DEVICES=1 python3 test_2gru_attn.py attn_highDic_lowTF/word_dict.txt attn_highDic_lowTF/encoder.pt attn_highDic_lowTF/decoder.pt test_input.txt attn_highDic_lowTF/test_output.txt 0 0
python3 test_2gru_attn.py attn_highDic_lowTF/word_dict.txt attn_highDic_lowTF/encoder.pt attn_highDic_lowTF/decoder.pt test_input.txt attn_highDic_lowTF/test_output_beam.txt 1 0
python3 test_2gru_attn.py attn_highDic_lowTF/word_dict.txt attn_highDic_lowTF/encoder.pt attn_highDic_lowTF/decoder.pt test_input.txt attn_highDic_lowTF/test_output_beam_norm.txt 1 1

