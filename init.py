from custom_model import ACCNNModel
import pickle

model = ACCNNModel(observation_space = [128,128,3], action_space = 5)
weights = model.get_weights()
with open('/root/model/init_model.pt', 'wb') as f:
    pickle.dump(weights, f)
