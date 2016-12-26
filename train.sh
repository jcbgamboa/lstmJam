#! /bin/bash

echo "Training with params --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=100 --n_layers=10 --n_itr=15000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=100 --n_layers=10 --n_itr=15000


echo "Training with params --learning_rate=0.001 --dropout=1.0 --tie_weights --batch_size=128 --size=100 --n_layers=10 --n_itr=1500"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=1.0 --tie_weights --batch_size=128 --size=100 --n_layers=10 --n_itr=20000


echo "Training with params --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=100 --n_layers=10 --n_itr=15000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=100 --n_layers=10 --n_itr=15000


echo "Training with params --learning_rate=0.001 --dropout=1.0 --batch_size=128 --size=100 --n_layers=10 --n_itr=20000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=1.0 --batch_size=128 --size=100 --n_layers=10 --n_itr=20000


echo "Training with params --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=200 --n_layers=20 --n_itr=20000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=200 --n_layers=20 --n_itr=20000


echo "Training with params --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=300 --n_layers=20 --n_itr=20000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --tie_weights --batch_size=128 --size=300 --n_layers=20 --n_itr=20000


echo "Training with params --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=200 --n_layers=20 --n_itr=20000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=200 --n_layers=20 --n_itr=20000


echo "Training with params --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=300 --n_layers=20 --n_itr=20000"
rm -rf chkpnts/*
python trainBatchNorm.py --learning_rate=0.001 --dropout=0.5 --batch_size=128 --size=300 --n_layers=20 --n_itr=20000
