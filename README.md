# TranFuzz - Setup and Usage Guide
TranFuzz is a black-box adversarial attack and defense framework that leverages domain adaptation and fuzzing techniques to generate robust adversarial examples for model evaluation and robust training.


## Reference 
Hao Li, Shanqing Guo, Peng Tang, Chengyu Hu, Zhenxiang Chen: TranFuzz: An Ensemble Black-Box Attack Framework Based on Domain Adaptation and Fuzzing. ICICS (1) 2021: 260-275

Adaptation of the following code for better usability :
https://github.com/lihaoSDU/ICICS2021/

# Prerequisites:

Python >= 3.8  
GPU recommended (CUDA supported)  
Tested on Ubuntu 20.04 


## Installation

    python3 -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt


## Execution

1. Train the target model

  ```py 
  python3 DSAN/train_target_model.py --model_name densenet --epochs 200 --data_path ./datasets/office31/webcam/ --dataset_target webcam --num_classes 31 --batch_size 8 --save_path ./DSAN/models
  ``` 

2. Train the source model using the DSAN method
  ```py
  python3 DSAN/DSAN.py --target_model DSAN/models/target_webcam_densenet.pt --target_name webcam --source_name amazon
  ```

3. Test common attacks
  ```py
  python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method <attack_type>
  ```
4. Test the predictions of a model

  ```py
  python3 test/predictions.py --target_path ./DSAN/models/target_webcam_densenet.pt --target_dataset webcam
  ```

5. Run the fuzzer on a source model trained with DSAN

  ```py
  python3 fuzz/fuzzer_main.py --input_data ./datasets/office31/amazon/test/ --output_dir ./fuzz/data/target_webcam_densenet/ --input_model ./model_resnet50_amazon_webcam.pth
  ```

6. Predict the accuracy/ misclassification rates using the fuzzed images

  ```py
  python3 test/predictions.py --target_path ./DSAN/models/target_webcam_densenet.pt --target_dataset webcam  --target_model densenet --fuzz True
  ```

7. Training of a robust madry model (same for FBF)

  ```py
  python3 defense/robust_training.py --dataset_target webcam --epochs 20 --model_target densenet --batch_size 64 --adv_trainer madry
  ```

8. Training of a TranFuzz robust model

  ```py
  python3 defense/robust_TranFuzz.py --dataset_target amazon --epochs 200 --model_target densenet --batch_size 64 --clean_training True
  ```


## Modifications Applied

1. In `fuzz/lib/Fuzzer.py`
Replace the loop on `img_path` with:

    ```dataset = self.data_loader.dataset  
    image_filenames = dataset.imgs  
    img_name = os.path.basename(image_filenames[idx][0])
2. IN `DSAN/DSAN.py`
- Fix next() calls:
  ```
  data_source, label_source = next(iter_source)
  data_target, _ = next(iter_target)
  ```

- Add `weights_only=False` when loading the model:
  ```
  torch.load(model_path, weights_only=False)
  ```

3. In `DSAN/train_target_model.py`
Remove the underscore variable:
  ```
    for batch_idx, (inputs, labels, _)
  ```

to 

    for batch_idx, (inputs, labels)


4. In `DSAN/utils/Weight.py`
Remove the hardcoded 65 value. Instead, use the appropriate value from utils.config.


## Warnings & Fixes


| File | Line | Warning | Suggested Fix |
|------|------|---------|---------------|
| `coverage/CoverageUpdate.py` | 39 | `SyntaxWarning: "is" with a literal` | Replace `is` with `==` in conditionals |
| `coverage/CoverageUpdate.py` | 39 | Repeated warning (appears twice) | Double-check both occurrences and correct both |
| `utils/mutators.py` | 38 | `UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow` | Convert list to a single `np.array()` before `torch.tensor()` |
| `lib/Fuzzer.py` | 130 | `UserWarning: To copy construct from a tensor...` | Use `source_tensor.clone().detach()` instead of `torch.tensor(source_tensor)` |

# Notes

slurm-201830.out = office31 webcam densenet target  
slurm-201834.out = office_home Product densenet target  
slurm-201836.out = office31 webcam densenet target again  
