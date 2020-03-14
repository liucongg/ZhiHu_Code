"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/3/13 21:34
"""
import jieba
from tf_idf import TF_IDF_Model


document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                 "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                 "我在微信上被骗了，请问被骗多少钱才可以立案？",
                 "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                 "有人走私两万元，怎么处置他？",
                 "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
document_list = [list(jieba.cut(doc)) for doc in document_list]
print(document_list)
# exit()
tf_idf_model = TF_IDF_Model(document_list)
print(tf_idf_model.documents_list)
print(tf_idf_model.documents_number)
print(tf_idf_model.tf)
print(tf_idf_model.idf)
query = "走私了两万元，在法律上应该怎么量刑？"
query = list(jieba.cut(query))
scores = tf_idf_model.get_documents_score(query)

print(scores)


