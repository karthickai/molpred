# Molecular Property Prediction using DeBerta

![Molecular Property Prediction Architecture](https://karthick.ai/assets/img/blog_molecular_property/mol_arch.svg)


## Read this Blog post for further explanation: https://karthick.ai/blog/2024/Molecular-Property-Prediction/


To create a DeBERTa tokenizer for Selfies notation. It will create DeBERTa tokenizer in the data directory.

```
python mol-tokenzier.py
```

To pretrain the DeBERTa model large corpus of Selfies notations. It will generate results directory with model checkpoint.
```
python pretrain.py
```

To fintue Pre trained DeBERTa model with Lipo dataset for Molecular Property prediction.

```
python fintune.py
```