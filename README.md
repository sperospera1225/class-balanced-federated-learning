# OnlineFedTuner Learning Process

- **Assumption:** An imbalanced dataset, DX, is collected in each round.

- **Client-side Training:** 
  - Each client trains on the DX dataset.
  - Generates class-specific loss graphs.
  - Sends the trained models back to the server.

- **Server-side Processing:** 
  - Averages the class-specific loss graphs from all clients.
  - Estimates the average amount of data needed to be supplemented for training.
  - Averages the models and sends them back to the clients.

- **Next Round Client-side Training:** 
  - Each client samples an amount of the newly collected DX dataset, as estimated by the server, for the next round of training.
