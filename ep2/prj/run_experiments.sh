echo "Experimentos sem NILC"
python ep2.py --dropout 0                     --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/lstm_dropout_000.txt
python ep2.py --dropout 0.25                  --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/lstm_dropout_025.txt
python ep2.py --dropout 0.5                   --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/lstm_dropout_050.txt
python ep2.py --dropout 0     --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/bidi_dropout_000.txt
python ep2.py --dropout 0.25  --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/bidi_dropout_025.txt 
python ep2.py --dropout 0.5   --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv > saved_logs/bidi_dropout_050.txt

echo "Experimentos com NILC"
python ep2.py --dropout 0                     --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/lstm_dropout_000_nilc.txt
python ep2.py --dropout 0.25                  --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/lstm_dropout_025_nilc.txt
python ep2.py --dropout 0.5                   --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/lstm_dropout_050_nilc.txt
python ep2.py --dropout 0     --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/bidi_dropout_000_nilc.txt
python ep2.py --dropout 0.25  --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/bidi_dropout_025_nilc.txt 
python ep2.py --dropout 0.5   --lstm_bidirect --b2w_path  ~/workspaces/usp/b2w-reviews01/B2W-Reviews01.csv --nilc_path ~/workspaces/usp/NILC/word2vec_200k.txt   > saved_logs/bidi_dropout_050_nilc.txt
