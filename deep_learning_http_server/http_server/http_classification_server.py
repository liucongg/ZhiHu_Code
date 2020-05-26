"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:00
"""
# gevent == 1.3a1
# flask == 0.12.2
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import wsgi
import json
import yaml
from ClassificationModel import ClassificationModel
import argparse


def start_sever(http_id, port, gpu_id, vocab_file, gpu_memory_fraction, model_path, max_seq_length):
    model = ClassificationModel()
    model.load_model(gpu_id, vocab_file, gpu_memory_fraction, model_path, max_seq_length)
    print("load model ending!")
    app = Flask(__name__)

    @app.route('/')
    def index():
        return "This is News Classification Model Server"

    @app.route('/news-classification', methods=['Get', 'POST'])
    def response_request():
        if request.method == 'POST':
            text = request.form.get('text')
        else:
            text = request.args.get('text')
        label, label_name = model.predict(text)
        d = {"label": str(label), "label_name": label_name}
        print(d)
        return json.dumps(d, ensure_ascii=False)

    server = wsgi.WSGIServer((str(http_id), port), app)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", type=str, default="news.yaml", help="yaml file")
    config = parser.parse_args()
    with open(config.yaml_file, "r") as yamlfile:
        cfg = yaml.load(yamlfile)

    http_id = cfg['hyperparameters']['http_id']
    port = cfg['hyperparameters']['port']
    gpu_id = cfg['hyperparameters']['gpu_id']
    gpu_memory_fraction = cfg['hyperparameters']['gpu_memory_fraction']
    max_seq_length = cfg['hyperparameters']['max_seq_length']

    model_path = cfg['path']['model_path']
    vocab_file = cfg['path']['vocab_file']
    start_sever(http_id, port, gpu_id, vocab_file, gpu_memory_fraction, model_path, max_seq_length)


if __name__ == "__main__":
    main()