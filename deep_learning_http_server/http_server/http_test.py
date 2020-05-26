"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:00
"""
import requests


def http_test(text):
    url = 'http://127.0.0.1:5555/news-classification'
    raw_data = {'text': text}
    res = requests.post(url, raw_data)
    result = res.json()
    return result


if __name__ == "__main__":
    text = "姚明在NBA打球，很强。"
    result = http_test(text)
    print(result["label_name"])