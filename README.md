# BlockCIrculantRNN

BlockCIrculantRNN (LSTM and GRU) using TensorFlow. This project implements the
block circulant matrix based training of RNN models, The code is based on clean
version of [this](https://github.com/zzw922cn/Automatic_Speech_Recognition)
project. For more details, please refer to the paper
```
@inproceedings{wang2018c,
  title={C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs},
  author={Wang, Shuo and Li, Zhe and Ding, Caiwen and Yuan, Bo and Qiu, Qinru and Wang, Yanzhi and Liang, Yun},
  booktitle={Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
  pages={11--20},
  year={2018},
  organization={ACM}
}
```
Other references
```
@article{li2018efficient,
  title={Efficient Recurrent Neural Networks using Structured Matrices in FPGAs},
  author={Li, Zhe and Wang, Shuo and Ding, Caiwen and Qiu, Qinru and Wang, Yanzhi and Liang, Yun},
  journal={arXiv preprint arXiv:1803.07661},
  year={2018}
}
```
Make sure you convert the TIMIT dataset from NIST format wav to RIFF format wavform
```bash
# Install sox
sudo apt-get install sox
# Use provided tools to convert TIMIT wav files
cd preprocessing
bash nist2wav.sh /xxx/xxx/TIMIT
```

To preprocess the TIMIT dataset, run
```bash
cd preprocessing
# Training set
python timit_preprocess_main.py \
--input_path=/xxx/xxx/TIMIT \
--output_path=/xxx/xxx/timit_preproc \
--level=phn \
--split=TRAIN

# Test set
python timit_preprocess_main.py \
--input_path=/xxx/xxx/TIMIT \
--output_path=/xxx/xxx/timit_preproc \
--level=phn \
--split=TEST
```

To Train the model, run
```bash
# Training a baseline
python timit_train_eval.py \
--input_data_dir=/xxx/xxx/timit_preproc \
--exp_dir=/xxx/xxx/timit_exp \
--level=phn \
--cell=LSTM \
--is_training=True

# Training block circulant model
python timit_train_eval.py \
--input_data_dir=/xxx/xxx/timit_preproc \
--exp_dir=/xxx/xxx/timit_exp \
--level=phn \
--cell=LSTM \
--partition_size=8 \
--is_training=True
```

To test the model, run
```bash
# Test a baseline
python timit_train_eval.py \
--input_data_dir=/xxx/xxx/timit_preproc \
--exp_dir=/xxx/xxx/timit_exp \
--level=phn \
--cell=LSTM \
--is_training=False \
--restore=True

# Test block circulant model
python timit_train_eval.py \
--input_data_dir=/xxx/xxx/timit_preproc \
--exp_dir=/xxx/xxx/timit_exp \
--level=phn \
--cell=LSTM \
--partition_size=8 \
--is_training=False \
--restore=True
```

For more argument options, check the code.
