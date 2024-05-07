import json
import pickle
from device import *

with open('config.json','r') as f:
    config = json.load(f)

#cuda random seed 설정
seed = config['Client']['random_seed']
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#config load
rnd = config['General']['round']
device = config['General']['device']
num_clients = config['General']['num_clients']
num_subsets = config['General']['num_subsets']
show_figure = config['General']['show_figure']

######################
#round0
server = Server(config)
clients = []


for n in range(num_clients):
    print("round : 0")
    clients.append(Client(config,n))
    clients[n].loadData(0) # id,rnd
    clients[n].trainOnSubsets(num_subsets)

server.estimate(show_figure=show_figure, rnd=0)

for r in range(1,rnd):
    print("round : ", str(r))
    for n in range(num_clients):
        clients[n].loadData(r)
        clients[n].trainOnEstimated()
    server.estimate(show_figure=show_figure, rnd=r)
