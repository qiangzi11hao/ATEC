一. 文件结构
dev version：
Antfin 
    --data			
	--data_all.csv		//data set, 102k
	--user_dict.txt		//分词词典，用于jieba
	--sgns.merge.word		//pretrained embedding
    --saved_model		//used for saved model weight
    --utils
	--__init__.py		//used for import langconv
	--langconv.py		//用于繁体->简体
	--zh_wiki.py		//同上
    --main.py		//used for preparation, train, and predict.
    --max_mag_embedding_model.py		//class of model
    --vocab.py		//class of vocab containing embedding matrix, word2id, padding...
    --readme.txt
How to run:
Run main.py, then will get vocab.data and the weights inside saved_models.

sgns.merge.word: https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw
	


