import os
import torch
import pickle
from model import *
from cnn import *
import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
import torch.optim as optim
from data import *
from operator import add

class Server(object):
    def __init__(self,config):
        super().__init__()
        self.communication_round = 0
        self.aggregation_type = config['Server']['aggregation']
        self.num_clients = config['General']['num_clients']
        self.budget = config['General']['budget']        
        self.global_model_src = config['Server']['src']
        self.global_fig_src = config['Server']['fig_dir']
        self.local_model_src = config['Client']['src']
        self.class_name = config['Data']['class_name']
        self.Lambda = config['Client']['Lambda']
        self.num_class = config['Data']['num_class']
        self.data_num_array = [0 for _ in range(self.num_class)]
        self.slice_num = [0 for _ in range(self.num_class)]
        self.loss_output = [0 for _ in range(self.num_class)]

    def estimate(self, show_figure, rnd):
        self.communication_round = rnd
        for client_id in range(self.num_clients):
            with open(self.global_model_src + '/data_num_array_'+str(client_id)+'.pickle', 'rb') as f:
                temp_num_array = pickle.load(f)
            # print(self.data_num_array, temp_num_array)
            self.data_num_array = [sum(x) for x in zip(self.data_num_array, temp_num_array)]
        
        self.data_num_array = [int(element/self.num_clients) for element in self.data_num_array]

        print("Data num array :", self.data_num_array)
        print('Data num array sum :', sum(self.data_num_array))
        for client_id in range(self.num_clients):
            with open(self.global_model_src + '/slice_num_'+str(client_id)+'.pickle', 'rb') as f:
                temp_slice_num = pickle.load(f)
                # print(temp_slice_num)
            if client_id == 0:
                self.slice_num = temp_slice_num
            else:
                self.slice_num = [[sum(x) for x in zip(a, b)] for a, b in zip(self.slice_num, temp_slice_num)]
        
        self.slice_num = [[int(x/self.num_clients) for x in sublist] for sublist in self.slice_num]

        # print(self.slice_num)

        for client_id in range(self.num_clients):
            with open(self.global_model_src + '/loss_output_'+str(client_id)+'.pickle', 'rb') as f:
                temp_loss_output = pickle.load(f)
                # print(temp_loss_output)
            if client_id == 0:
                self.loss_output = temp_loss_output
            else:
                self.loss_output = [[sum(x) for x in zip(a, b)] for a, b in zip(self.loss_output, temp_loss_output)]
        
        self.loss_output = [[x / self.num_clients for x in sublist] for sublist in self.loss_output]
        # print(self.loss_output)
                
        self.aggregation() 

        num_examples = self.one_shot(show_figure)
        print('Num examples: ', num_examples)

        with open(self.global_model_src + '/num_examples.pickle', 'wb') as f:
            pickle.dump(num_examples, f)
        

    def aggregation(self):
        print('global model aggregation')
        if self.aggregation_type == "FedAvg":
            global_model = self.loadModel(0)
            for param in global_model.parameters():
                param.data.zero_()

            for client_id in range(self.num_clients):
                local_model = self.loadModel(client_id)
                for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                    global_param.data.add_(local_param.data)
            
            for param in global_model.parameters():
                param.data.div_(self.num_clients)
            self.saveModel(global_model)
            
        return global_model
    
    def saveModel(self,global_model):
        torch.save(global_model, self.global_model_src+'/global_model.p')

    def loadModel(self, client_id):
        model_path = self.local_model_src+'/local_'+str(client_id)
        model = torch.load(model_path)
        return model

    def one_shot(self, show_figure):
        print('one shot')
        A, B, estimate_loss = self.fit_learning_curve(show_figure)
        print(A, B, estimate_loss)
        return self.op_func(A, B, estimate_loss)
        
    def fit_learning_curve(self, show_figure):
        print('fit learning curve')
        def weight_list(weight):
            w_list = []
            for i in weight:
                w_list.append(1/(i**0.5))
            return w_list        

        def power_law(x, a, b):
            return (b*x**(-a))
        
        A = []
        B = []
        estimate_loss = []
        
        for i in range(self.num_class):
            xdata = np.linspace(self.slice_num[i][0], self.slice_num[i][-1], 1000)
            sigma = weight_list(self.slice_num[i])
            print(self.slice_num[i], type(self.slice_num[i]))
            print(self.loss_output[i], type(self.loss_output[i]))
            popt, pcov = curve_fit(power_law, self.slice_num[i], self.loss_output[i], sigma=sigma, absolute_sigma=True)
            
            A.append(-popt[0])
            B.append(popt[1])
            estimate_loss.append(popt[1] * (self.data_num_array[i] ** (-popt[0])))
            
            fig_folder_path = os.path.join(self.global_fig_src, self.class_name[i])

            # 해당 경로에 폴더가 존재하지 않으면 생성합니다.
            if not os.path.exists(fig_folder_path):
                os.makedirs(fig_folder_path)

            if show_figure == True:
                plt.figure(1, figsize=(12,8))
                # plt.plot(self.slice_num[i], self.loss_output[i], 'o-', linewidth=1.0, markersize=4, label=self.class_name[i])
                plt.plot(xdata, power_law(xdata, *popt), linewidth=2.0, label='$y={%0.3f}x^{-{%0.3f}}$' % (popt[1], popt[0]))

                plt.tick_params(labelsize=20)
                plt.xlabel('Number of training examples', fontsize=25)
                plt.ylabel('Validation Loss', fontsize=25)
                plt.legend(prop={'size':25})

                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(fig_folder_path, f'{self.communication_round}_lossgraph.png'))
                plt.close()
        return A, B, estimate_loss

    def op_func(self, A, B, estimate_loss):
        print('convex optimization')    
        print('changed')
        
        x = cp.Variable(self.num_class, integer=True)
        for i in range(self.num_class):
            loss = cp.multiply(B[i], cp.power((x[i]+self.data_num_array[i]), A[i]))
            counter_loss = (np.sum(estimate_loss) - estimate_loss[i]) / (self.num_class - 1)
            if i==0:
                ob_func = loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)
            else:
                ob_func += loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)

        cost_func = [1] * self.num_class
        constraints = [cp.sum(cp.multiply(x, cost_func)) <= self.budget] + [x>=0]
        objective = cp.Minimize(ob_func)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="ECOS_BB")
        print(self.num_class, self.data_num_array, self.budget, self.Lambda)
        print(prob.status, prob.value)
        print('x value is', x.value)
        return np.add(x.value, 0.5).astype(int)

    
    

class Client(object):
    def __init__(self, config, id):
        super().__init__()
        self.config = config
        self.learning_rate = config['Client']['learning_rate']
        self.epoch = config['Client']['epoch']
        self.num_clients = config['General']['num_clients']
        self.batch_size = int(config['Client']['batch_size'])

        self.num_examples = None
        self.num_iter = config['General']['num_iter']
        self.budget = config['General']['budget']
        self.Lambda = config['Client']['Lambda']
        self.num_class = config['Data']['num_class']
        self.class_name = config['Data']['class_name']
        self.imbalance_ratio = 1
        self.ID = id

        self.loss_output=[]
        self.slice_num=[]
        self.val_data_dict = []
        self.add_data_dict = []
        self.device = config['General']['device']
        self.local_src = config['Client']['src']

        self.model = Resnet50()
        self.criterion = nn.MSELoss().to(self.device)

    def loadData(self,rnd):
        print("[Clients]["+str(self.ID)+"] : "+"data load")
        # load data
        fashionMNIST = FashionMNIST(self.config)
        self.train, self.val, self.data_num_array, self.val_data_dict = fashionMNIST.initialDataLoad(self.ID,rnd)
        slice_desc = []
        for i in range(self.num_class):
            slice_desc.append('Slice: %s, Number of data: %d' % (self.class_name[i], self.data_num_array[i]))
        with open(self.local_src + '/data_num_array_'+str(self.ID)+'.pickle','wb') as f:
            pickle.dump(self.data_num_array,f)

    def check_num(self, labels):
        """ Checks the number of data per each slice 
        Args:
            labels: Array that contains only label
        """
        
        slice_num = dict()
        for j in range(self.num_class):
            idx = np.argmax(labels, axis=1) == j
            slice_num[j] = len(labels[idx])
            
        return slice_num

    def trainOnSubsets(self, num_subsets):
        train_x, train_y = self.train
        val_x, val_y = self.val
        
        initial_subset = 200
        subsets = initial_subset + np.arange(0, num_subsets) * (len(train_x) - initial_subset)/ (num_subsets-1)
        subsets = [int(i) for i in subsets]

        for i in range(self.num_class):
            self.loss_output.append([0] * num_subsets)
            self.slice_num.append([])

        for k in range(num_subsets):
            print('>>>>>>>> subssets:'+str(k))
            for i in range(self.num_iter):
                print('>>>> iter:'+str(i))
                model = self.model
                model = model.to(self.device)
                
                optimizer = optim.SGD(model.parameters(),lr=self.learning_rate)
                
                min_loss = 100    
                loss_dict = {}
                slice_num = self.check_num(train_y[:subsets[k]]) # 추후 수정 필요
                for e in range(self.epoch):
                    print('>> epoch:'+str(e))
                    model.train()
                    for b in range(0,subsets[k],self.batch_size): # 추후 배치부분 custom dataset + dataloader로 수정
                        be = min(b+self.batch_size,subsets[k])
                        x =train_x[b:be].to(self.device)
                        y = train_y[b:be].to(self.device)
                        results = model(x)
                        loss = self.criterion(results, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    with torch.no_grad():
                        model.eval()
                        val_size = len(val_y)
                        val_loss=0.0
                        for b in range(0,val_size,self.batch_size):
                            be = min(b+self.batch_size,val_size)
                            x = val_x[b:be].to(self.device)
                            y = val_y[b:be].to(self.device)
                            results = model(x)
                            val_loss += self.criterion(results, y)
                            val_loss /= len(y)

                        if min_loss > val_loss:
                            min_loss = val_loss
                            for j in range(self.num_class):
                                val_size = len(self.val_data_dict[j][1])
                                eval_loss = 0.0
                                for b in range(0,val_size,self.batch_size):
                                    be = min(b+self.batch_size,val_size)
                                    x = self.val_data_dict[j][0][b:be].to(self.device)
                                    y = self.val_data_dict[j][1][b:be].to(self.device)
                                    eval_loss += self.criterion(model(x),y)/len(y)
                                loss_dict[j] = eval_loss
                
                for j in range(self.num_class):
                    self.loss_output[j][k] += (loss_dict[j] / self.num_iter).cpu().item()
                    if i == 0:
                        self.slice_num[j].append(slice_num[j])
        
        with open(self.local_src + '/slice_num_'+str(self.ID)+'.pickle','wb') as f:
            pickle.dump(self.slice_num, f)
        with open(self.local_src + '/loss_output_'+str(self.ID)+'.pickle','wb') as f:
            pickle.dump(self.loss_output, f)

        self.imbalance_ratio = self.get_imbalance_ratio(self.data_num_array)
        self.saveModel(model)

    def collectData(self,train_x,train_y,data_count): # num_example에 맞게 데이터 수정
        def shuffle(data, label):
            shuffle_idx = np.arange(len(data))
            np.random.shuffle(shuffle_idx)
            data = data[shuffle_idx]
            label = label[shuffle_idx]
            return data, label
        
        if data_count !=0:
            new_data_x = train_x
            new_data_y = train_y
            train_x, train_y = self.train

        for i in range(self.num_class):
            idx = np.argmax(train_y,axis=1)==i
            if data_count ==0 and i==0:
                new_data_x = train_x[idx][:self.num_examples[i]]
                new_data_y = train_y[idx][:self.num_examples[i]]
            else:
                new_data_x = torch.cat((new_data_x,train_x[idx][:self.num_examples[i]]),axis=0)
                new_data_y = torch.cat((new_data_y,train_y[idx][:self.num_examples[i]]),axis=0)
        new_data_x, new_data_y = shuffle(new_data_x,new_data_y)
        return new_data_x, new_data_y


    def trainOnEstimated(self):
        train_x, train_y = self.train
        val_x, val_y = self.val
        self.T = 1
        budget = self.budget
        self.cost_func = [1]*self.num_class


        # 다음 수집 데이터 정보 갱신
        with open(self.local_src + '/num_examples.pickle','rb') as f:
            self.num_examples = pickle.load(f)
        
        pre_loss_output = self.loss_output
        self.loss_output=[]
        self.slice_num=[]
        for i in range(self.num_class):
            self.loss_output.append(0)
            self.slice_num.append([])

        data_count = 0
        while budget > 0 :
            after_imbalance_ratio = self.get_imbalance_ratio(self.data_num_array + self.num_examples)
            if abs(self.imbalance_ratio-after_imbalance_ratio) > self.T:
                target_ratio = self.imbalance_ratio + self.T*np.sign(self.imbalance_ratio-after_imbalance_ratio)
                change_ratio = self.get_change_ratio(self.data_num_array, self.num_examples, target_ratio)
                self.num_examples = [int(self.num_examples[i]*change_ratio) for i in range(self.num_class)]
            train_x, train_y = self.collectData(train_x,train_y,data_count)
            for i in range(self.num_iter):
                print('>>>> iter:'+str(i))
                model = self.loadModel()
                model = model.to(self.device)
                
                optimizer = optim.SGD(model.parameters(),lr=self.learning_rate)
                
                min_loss = 100    
                loss_dict = {}
                slice_num = self.check_num(train_y)
                for e in range(self.epoch):
                    print('>> epoch:'+str(e))
                    model.train()
                    for b in range(0,len(train_y),self.batch_size): # 추후 배치부분 custom dataset + dataloader로 수정
                        be = min(b+self.batch_size,len(train_y))
                        x =train_x[b:be].to(self.device)
                        y = train_y[b:be].to(self.device)
                        results = model(x)
                        loss = self.criterion(results, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    with torch.no_grad():
                        model.eval()
                        val_size = len(val_y)
                        val_loss=0.0
                        for b in range(0,val_size,self.batch_size):
                            be = min(b+self.batch_size,val_size)
                            x = val_x[b:be].to(self.device)
                            y = val_y[b:be].to(self.device)
                            results = model(x)
                            val_loss += self.criterion(results, y)
                            val_loss /= len(y)

                        if min_loss > val_loss:
                            min_loss = val_loss
                            for j in range(self.num_class):
                                val_size = len(self.val_data_dict[j][1])
                                eval_loss = 0.0
                                for b in range(0,val_size,self.batch_size):
                                    be = min(b+self.batch_size,val_size)
                                    x = self.val_data_dict[j][0][b:be].to(self.device)
                                    y = self.val_data_dict[j][1][b:be].to(self.device)
                                    eval_loss += self.criterion(model(x),y)/len(y)
                                loss_dict[j] = eval_loss
                
                for j in range(self.num_class):
                    self.loss_output[j] += (loss_dict[j] / self.num_iter).cpu().item()
                    if i == 0:
                        self.slice_num[j].append(slice_num[j])
                        
                
                self.data_num_array = [self.data_num_array[i] + self.num_examples[i] for i in range(self.num_class)]  
                budget = budget - np.sum(np.add(np.multiply(self.num_examples, self.cost_func), 0.5).astype(int))
                self.increase_limit('aggressive')
                self.imbalance_ratio = after_imbalance_ratio
                print("======= Collect Data =======")
                print(self.num_examples)
                print("Total Cost: %s, Remaining Budget: %s" 
                  % (np.sum(np.add(np.multiply(self.num_examples, self.cost_func), 0.5).astype(int)), budget))
            
            print("======= Performance =======")
            print("Strategy: %s, C: %s, Budget: %s" % ('aggressive', self.Lambda, budget))
            print("Loss: %.5f, Average EER: %.5f, Max EER: %.5f\n" % tuple(self.show_performance()))
            data_count+=1
            #예외코드
            buget = 0
        for j in range(self.num_class):
            self.loss_output[j] = pre_loss_output[j][1:] + self.loss_output[j]

        with open(self.local_src + '/slice_num_'+str(self.ID)+'.pickle','wb') as f:
            pickle.dump(self.slice_num, f)
        with open(self.local_src + '/loss_output_'+str(self.ID)+'.pickle','wb') as f:
            pickle.dump(self.loss_output, f)
        self.imbalance_ratio = self.get_imbalance_ratio(self.data_num_array)
        self.saveModel(model)

    def show_performance(self):
        """ Average validation loss, Average equalized error rate(Avg. EER), Maximum equalized error rate (Max. EER) """
        
        final_loss = []
        num = 0
        max_eer = 0
        avg_eer =0
        
        for i in range(self.num_class):
            final_loss.append(self.loss_output[i])
            
        for i in range(self.num_class):
            diff_eer = ((np.sum(final_loss) - final_loss[i]) / (self.num_class-1) - final_loss[i]) * (-1)
            if diff_eer > 0:
                if max_eer < diff_eer:
                    max_eer = diff_eer
                
                avg_eer += diff_eer
                num += 1
                
        avg_eer = avg_eer / num
        return np.average(final_loss), avg_eer, max_eer 

    def increase_limit(self, strategy):
        
        if strategy == "aggressive":
            self.T = self.T * 2
        elif strategy == "linear":
            self.T = self.T + 1

    def loadModel(self):
        model = torch.load(self.local_src+'/global_model.p')
        return model
        
    def saveModel(self,local_model):
        torch.save(local_model, self.local_src+'/local_'+str(self.ID))
        
    def get_imbalance_ratio(self, data_array):
        
        return max(data_array) / min(data_array)

    def get_change_ratio(self, data_array, num_examples, target_ratio):

        def F(x, num, add, target):
            func1 = max([int(num_examples[i]*x) + num[i] for i in range(self.num_class)])
            func2 = min([int(num_examples[i]*x) + num[i] for i in range(self.num_class)])
            return func1 - target * func2

        ratio = scipy.optimize.fsolve(F, x0=(0.5), args=(data_array, num_examples, target_ratio))
        if ratio < 0:
            ratio =  scipy.optimize.fsolve(F, x0=(1.0), args=(data_array, num_examples, target_ratio))
        elif ratio > 1:
            ratio = scipy.optimize.fsolve(F, x0=(0.01), args=(data_array, num_examples, target_ratio))

        return ratio
