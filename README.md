# tic-tac-toe
AI Neural Network Plays Tic-Tac-Toe

Implementation code is located in [main.py](https://github.com/jkaldon/tic-tac-toe/blob/master/main.py).

```
$ pip3 install tensorflow keras
$ python3 main.py
Using TensorFlow backend.
Load previously trained NN? (Y/n)
Colocations handled automatically by placer.
2019-03-11 22:56:42.456307: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2101000000 Hz
2019-03-11 22:56:42.462495: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x1d75f60 executing computations on platform Host. Devices:
2019-03-11 22:56:42.462619: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
Loaded model from disk
results = [wins, ties, losses]
[2387  613    0]
```

The properly trained Neural Network is named `model-sigmoid-prelu-10x.(json|h5)`.
