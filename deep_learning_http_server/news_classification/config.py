"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:11
"""

import os
import tensorflow as tf
from main import train, save_pb
from prepro import prepro
flags = tf.flags

home = "./"
train_file = os.path.join(home, "data", "news_train_data.json")
dev_file = os.path.join(home, "data", "news_dev_data.json")


model_name = "new_classification_bert"
dir_name = os.path.join(home, model_name)

if not os.path.exists(os.path.join(os.getcwd(),dir_name)):
    os.mkdir(os.path.join(os.getcwd(), dir_name))

target_dir = dir_name
save_dir = os.path.join(dir_name, "model")


train_eval = os.path.join(target_dir, "train_data.json")
dev_eval = os.path.join(target_dir, "dev_data.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")
flags.DEFINE_string("bert_config_file",
                    "./data/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json",
                    "The config json file corresponding to the pre-trained BERT model. ")
flags.DEFINE_string("vocab_file",
                    "./data/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")

flags.DEFINE_string("train_eval", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval", dev_eval, "Out file for dev eval")

flags.DEFINE_integer("max_seq_length", 512, "")

flags.DEFINE_string("init_checkpoint",
                    "./data/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt",
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer("num_warmup_steps", 200, "num warmup steps")
flags.DEFINE_integer("num_train_steps", 0, "num warmup steps")
flags.DEFINE_string("gpu_id", "1", "num warmup steps")
flags.DEFINE_integer("batch_size", 10, "Batch size")
flags.DEFINE_integer("print_pre_batch", 200, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("num_epochs", 20, "Number of batches to evaluate the model")
flags.DEFINE_bool("use_one_hot_embeddings", False, "")
flags.DEFINE_float("learning_rate", 3e-5, "Learning rate")
flags.DEFINE_integer("early_stop", 20, "Checkpoints for early stop")

def main(_):
    config = flags.FLAGS
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "save_pb":
        save_pb(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
