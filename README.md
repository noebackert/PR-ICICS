1. run run.sh of DSAN
2. run run.sh of fuzz



Ajout de ce code dans lib/Fuzzer.py à la place de la boucle sur img_path qui n'existe pas

``` 
dataset = self.data_loader.dataset
image_filenames = dataset.imgs
img_name = os.path.basename(image_filenames[idx][0])  
```


Correction de next() dans DSAN.py

    data_source, label_source = next(iter_source)
    data_target, _ = next(iter_target)



Ajout de weights_only=False dans DSAN.py torch.load()


Enlever '_' dans train_target_model

    for batch_idx, (inputs, labels, _)


Dans Weights, enlever le 65 hardcodé qu génère une erreur lors de l'entrainement DSAN pour office31, ajouté celui de utils.config



Reference: 

Hao Li, Shanqing Guo, Peng Tang, Chengyu Hu, Zhenxiang Chen: TranFuzz: An Ensemble Black-Box Attack Framework Based on Domain Adaptation and Fuzzing. ICICS (1) 2021: 260-275
