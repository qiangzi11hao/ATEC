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
	

本项目中由于采用的word2vec与之前的不同，所以运行时需要将sgns.merge.word改成sgns.merge.char，经过之前的一次测试，两者差别不大，没有进行深入的比较。

model的框架主要参考的QA-LSTM，通过将embedding后的vector，先进行biLSTM layer进行编码，然后通过一个multi-head conv层（kernel size =【2,3,5】），再进行max-pooling，将三层结果concate，直接送入dense网络。

改进idea：
	1.biLSTM可以在加一层，然后采用highway network扩展语义
	2.在max-pooling后加入人工特征，进一步进行提取
	3.最后加match layer
	 
