import yaml
from pyt_ex import neural_net as ann

with open('config/Input.yaml') as File:
        dic = yaml.load(File, Loader=yaml.FullLoader)

for i in dic:
    dnn = ann.dynamic_nn(dic[i])
    dnn.show() 