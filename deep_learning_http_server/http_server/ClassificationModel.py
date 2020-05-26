"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:00
"""
import bert_tokenization
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os


class ClassificationModel(object):
    def __init__(self):
        self.tokenizer = None
        self.sess = None
        self.is_train = None
        self.input_ids = None
        self.input_mask = None
        self.segment_ids = None
        self.predictions = None
        self.max_seq_length = None
        self.label_dict = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']

    def load_model(self, gpu_id, vocab_file, gpu_memory_fraction, model_path, max_seq_length):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        self.tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)
        with gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name="")

        self.sess.run(tf.global_variables_initializer())
        self.is_train = self.sess.graph.get_tensor_by_name("input/is_train:0")
        self.input_ids = self.sess.graph.get_tensor_by_name("input/input_ids:0")
        self.input_mask = self.sess.graph.get_tensor_by_name("input/input_mask:0")
        self.segment_ids = self.sess.graph.get_tensor_by_name("input/segment_ids:0")
        self.predictions = self.sess.graph.get_tensor_by_name("output_layer/predictions:0")
        self.max_seq_length = max_seq_length


    def convert_fearture(self, text):
        max_seq_length = self.max_seq_length
        max_length_context = max_seq_length - 2

        content_token = self.tokenizer.tokenize(text)
        if len(content_token) > max_length_context:
            content_token = content_token[:max_length_context]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in content_token:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        segment_ids = np.array(segment_ids)
        return input_ids, input_mask, segment_ids

    def predict(self, text):
        input_ids_temp, input_mask_temp, segment_ids_temp = self.convert_fearture(text)
        feed = {self.is_train: False,
                self.input_ids: input_ids_temp.reshape(1, self.max_seq_length),
                self.input_mask: input_mask_temp.reshape(1, self.max_seq_length),
                self.segment_ids: segment_ids_temp.reshape(1, self.max_seq_length)}
        [label] = self.sess.run([self.predictions], feed)
        label_name = self.label_dict[label[0]]
        return label[0], label_name


if __name__ == "__main__":
    model = ClassificationModel()
    gpu_id = "1"
    vocab_file = "../news_classification/data/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt"
    gpu_memory_fraction = 0.5
    model_path = "../news_classification/new_classification_bert/model/bert_model.pb"
    max_seq_length = 512
    model.load_model(gpu_id, vocab_file, gpu_memory_fraction, model_path, max_seq_length)

    # text = "湖人这800万真不亏 国王首次进攻就给湖人一个下马威，东特-格林埋伏在右侧45度角投中3分球，但加索尔马上有所反应，拉到中距离面对两名球员的骚扰，果断中投得手。"
    text = "《红派：游击队》采用全粉碎破坏引擎 玩过《红色派系》、《红色派系2》的玩家，应该还记得可以直接爽快的破坏环境中的物体，任意破坏阻挡去路的墙壁，完全无视环境的限制。"
    label, label_name = model.predict(text)
    print(label)
    print(label_name)



