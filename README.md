cat > README.md <<'MD'
# Boston Housing - MLOps Assignment 1

This repo contains a minimal MLOps pipeline for the Boston housing dataset:
- dtree branch: DecisionTreeRegressor training (train.py)
- kernelridge branch: KernelRidge training (train2.py) + GitHub Actions CI

Files:
- misc.py : reusable pipeline functions
- train.py : Decision Tree script
- train2.py : Kernel Ridge script
- requirements.txt : dependencies
- .github/workflows/ci.yml : CI workflow

Run locally:
1. create conda env: conda create -n boston-ml python=3.10 -y
2. activate and install: conda activate boston-ml; pip install -r requirements.txt
3. Run DT: python train.py
4. Run KR: python train2.py --kernel linear --alpha 1.0

Note: Dataset is loaded from http://lib.stat.cmu.edu/datasets/boston as required (sklearn.load_boston is deprecated).
MD
