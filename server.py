from sklearn.neural_network import MLPClassifier as nn
import json
import numpy as np
class NeuralNetworkHandler:
    def __init__(self):
        self.nn_model = nn(hidden_layer_sizes=(100,), max_iter=1000)  # Initialize with your desired parameters

    def train(self, train_data):
        X_train = [item['features'] for item in train_data]  # Assuming each item has 'features' and 'label'
        y_train = [item['label'] for item in train_data]
        self.nn_model.fit(X_train, y_train)

    def predict(self, image_data):
        return self.nn_model.predict([image_data])[0]  # Assuming image_data is in the same format as training features

    def save(self, file_path="neural_network.json"):
        model_params = {
            "coefs": [coef.tolist() for coef in self.nn_model.coefs_],
            "intercepts": [intercept.tolist() for intercept in self.nn_model.intercepts_]
        }
        with open(file_path, 'w') as f:
            json.dump(model_params, f)

    def load(self, file_path="neural_network.json"):
        with open(file_path, 'r') as f:
            model_params = json.load(f)
        self.nn_model.coefs_ = [np.array(coef) for coef in model_params['coefs']]
        self.nn_model.intercepts_ = [np.array(intercept) for intercept in model_params['intercepts']]

# Example usage in a server context
def do_POST(s):
    response_code = 200
    response = ""
    var_len = int(s.headers.get('Content-Length'))
    content = s.rfile.read(var_len)
    payload = json.loads(content)

    nn_handler = NeuralNetworkHandler()

    if payload.get('train'):
        nn_handler.train(payload['trainArray'])
        nn_handler.save()
    elif payload.get('predict'):
        try:
            result = nn_handler.predict(payload['image'])
            response = {
                "type": "test",
                "result": result
            }
        except Exception as e:
            response_code = 500
            response = {"error": str(e)}
    else:
        response_code = 400

    s.send_response(response_code)
    s.send_header("Content-type", "application/json")
    s.send_header("Access-Control-Allow-Origin", "*")
    s.end_headers()

    if response:
        s.wfile.write(json.dumps(response).encode('utf-8'))
    return
