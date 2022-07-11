# Aneja-Lab-Public-Adversarial-Detection

In this project, we will investigate the robustness of adversarial image detection models by testing their performance on adversarial examples from various medical imaging datasets. 

# Code Flow
```
- configs.py # Designates parameters for adversarial attack and training experiments
- utils.py # Defines useful functions used in experiments
- data_loader.py # Loads in desired training and test set, preprocesses data
- data_generator.py # Performs data augmentations (flips, rotations) on medical datasets
- models.py # Defines classification model (VGG-16 model)
- detector_models.py # Defines detection models
- train.py # Trains model on data
- attacks.py # Creates adversarial attacks
- main.py # Applies attack on desired dataset and evaluates model accuracy on adversarial examples
- adv_trainer.py # Applies PGD-based adversarial training to models
- detect.py # Creates and trains adversarial detection models and evaluates their efficacy against adversarial images
- compare.py # compares adversarial detection, adversarial training, and combination of the two methods on improvement of classification accuracy
 ```

# How to Run
1. Clone Repository from Github
2. Install necessary dependencies ```python pip3 install -r Requirements.txt```
3. Edit ```configs.py``` to customize parameters 
4. Run ```data_generator.py``` to augment medical datasets
5. Run ``` train.py ``` to train DNN models
6. Run ``` main.py ``` for adversarial attacking experiments
7. Run ```detect.py``` for adversarial detection experiments
8. Run ```adv_trainer.py``` for adversarial training experiments
9. Run ```compare.py``` for comparison experiments
