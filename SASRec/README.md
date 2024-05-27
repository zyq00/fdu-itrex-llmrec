update, with few lines of manually initialization code added, it converges as fast as tf version. BTW, I strongly recommend checking issues of the repo from time to time for knowing new updates and details :)

---

update: a pretrained model added, pls run the command as below to test its performance(current perf still not as good as paper's reported results after trained more epochs, maybe due to leaky causual attention weights issue got fixed by using PyTorch 1.6's MultiHeadAttention, pls help identifying the root cause if you are interested):

```

modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, executable by:
训练启动代码
python main.py --device=cuda --dataset=douban --train_dir=default --maxlen=2 --num_epochs=10 --inference_only=true --state_dict_path='douban_default/SASRec.epoch=10.lr=0.001.layer=2.head=1.hidden=50.maxlen=2.pth'
评估模式 需要把neural chat server先开起来
python main.py --device=cuda --dataset=douban --train_dir=default --maxlen=50 --num_epochs=30 --inference_only=true --state_dict_path='douban_default/SASRec.epoch=30.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth'

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
