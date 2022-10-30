.PHONY: train_vaeflow train_glf

train_vaeflow:
	pipenv run python3 train_vaeflow.py

train_glf:
	pipenv run python3 train_glf.py
