# Class Balanced Online Federated Learning

## Training procedures
- **Assumption:** An imbalanced dataset, $D_x$, is collected in each round.

- **Client-side Training:** 
  - Each client trains on the $D_x$.
  - Generates class-specific learning curves(loss graphs).
  - Sends the trained models back to the server.

- **Server-side Processing:** 
  - Averages the class-specific loss graphs from all clients.
  - Estimates the average amount of data needed to be supplemented for training.
  - Averages the models and sends them back to the clients.

- **Next Round Client-side Training:** 
  - Each client samples an amount of the newly collected $D_x$, as estimated by the server, for the next round of training.

The detailed algorithm is as follows:

<img width="450" alt="Screenshot 2023-11-24 at 1 44 27 PM" src="https://github.com/sperospera1225/selective_data_federated_learning/assets/67995592/353bc6d2-eb69-4610-87af-df9b600dc660">

## Estimated learning curves 
<img width="1203" alt="image" src="https://github.com/sperospera1225/class-balanced-federated-learning/assets/67995592/218f02e6-10d3-489a-a06a-65a77c4bddbe">

## Datasets
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
