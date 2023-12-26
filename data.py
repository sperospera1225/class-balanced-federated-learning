import random
import torch
from torchvision import datasets, transforms
import numpy as np

class FashionMNIST(object):
    def __init__(self,config):
        self.config = config
        self.src = config['Data']['src']
        self.val_data_num = config['General']['val_data_num']
        def toFloat(tensor):
            return tensor.float()
        self.toFloat = toFloat
        self.transform_x = transforms.Compose([
                                        transforms.Lambda(toFloat),
                                        transforms.Resize((224, 224)),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        self.transform_y = transforms.Compose([
                                        transforms.Lambda(toFloat),
        ])

    def shuffle(self, data, label):
        shuffle_idx = np.arange(len(data))
        np.random.shuffle(shuffle_idx)
        data = data[shuffle_idx]
        label = label[shuffle_idx]
        return data, label
    
    #def toCategorical(self,label,num_class):
    #    return torch.eye(num_classes)[label]

    def initialDataLoad(self,client_id,rnd):
        random.seed(hash((client_id,rnd)))
        if rnd ==0:
            self.batch_size = self.config['General']['initial_budget']
        else:
            self.batch_size = self.config['General']['budget']
        # Load Fashion MNIST data
        fashion_mnist_train = datasets.FashionMNIST(root=self.src, train=True, download=True)
        #fashion_mnist_test = datasets.FashionMNIST(root='./datasets', train=False, download=True)
        num_class = len(fashion_mnist_train.classes)

        train_x = fashion_mnist_train.data
        train_y = fashion_mnist_train.targets
        #test_x = fashion_mnist_test.data
        #test_y = fashion_mnist_train.targets

        train_x = train_x.unsqueeze(axis=1)
        #test_x = test_x.unsqueeze(axis=1)
        train_y = torch.nn.functional.one_hot(train_y,num_class)
        #test_y = torch.nn.functional.one_hot(test_y,num_class)

        initial_data_array=[]
        val_data_dict = []
        add_data_dict = []

        val_data_num = self.val_data_num
        class_idx = list(range(num_class))
        random.shuffle(class_idx)
        
        for i in range(num_class):
            if self.batch_size==2000:
                v=400
            else:
                v =400
            data_num = int(v / (num_class - class_idx[i]) ** 0.5) # 400은 buget에 dependent한 값
            if i==(num_class-1):
                if sum(initial_data_array)+data_num > self.batch_size:
                    data_num -= (sum(initial_data_array)+data_num - self.batch_size)
            initial_data_array.append(data_num)
            idx = np.argmax(train_y,axis=1) == i
            
            val_data_dict.append((self.transform_x(train_x[idx][data_num:data_num+val_data_num]), self.transform_y(train_y[idx][data_num:data_num+val_data_num])))
            #add_data_dict.append((train_x[idx][data_num+val_data_num:], train_y[idx][data_num+val_data_num:]))
            
            if i == 0:
                train_data = self.transform_x(train_x[idx][:data_num])
                train_label = self.transform_y(train_y[idx][:data_num])
                val_data = self.transform_x(train_x[idx][data_num:data_num+val_data_num])
                val_label = self.transform_y(train_y[idx][data_num:data_num+val_data_num])
            else:
                train_data = torch.cat((train_data, self.transform_x(train_x[idx][:data_num])), axis=0)
                train_label = torch.cat((train_label, self.transform_y(train_y[idx][:data_num])), axis=0) 
                val_data = torch.cat((val_data, self.transform_x(train_x[idx][data_num:data_num+val_data_num])), axis=0)
                val_label = torch.cat((val_label, self.transform_y(train_y[idx][data_num:data_num+val_data_num])), axis=0)
        
        train_data, train_label = self.shuffle(train_data,train_label)
        train_data = train_data[:self.batch_size]
        train_label = train_label[:self.batch_size]
        return (train_data, train_label), (val_data, val_label), initial_data_array, val_data_dict#, add_data_dict
    

if __name__ == '__main__':
    import json
    with open('config.json','r') as f:
        config = json.load(f)
    fashionMNIST = FashionMNIST(config)
    fashionMNIST.initialDataLoad(1,1)