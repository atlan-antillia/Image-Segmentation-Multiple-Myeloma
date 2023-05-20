rd /s /q  .\basnet_eval
rd /s /q  .\basnet_models
python ./TensorflowUNetMultipleMyelomaTrainer.py train_eval_infer_basnet_hybrid_loss.config

