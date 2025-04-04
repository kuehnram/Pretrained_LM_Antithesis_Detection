# Using Pre-trained Language Models in an End-to-End Pipeline for Antithesis Detection

This is additional material for the paper "Using Pre-trained Language Models in an
End-to-End Pipeline for Antithesis Detection."
Our goal is to identify the rhetorical figures antithesis in a German COVID-19 skeptics dataset by using pre-trained language models. The best results were achieved with the German ELECTRA model.

Parallelism_detection.py
This file contains the code to extract parallel phrases from the original dataset. Only if syntactic parallelism is present, the texts is a candidate for antithesis.

PosTags.py
Necessary file for the parallelism detection based on POS tags.


Antithesis_detection.py
This file contains the code to identify antitheses in the dataset. We also included the two augmentation techniques described in the paper.


RawDataset.csv
Original raw dataset.

antithesis_phrases_annotated_refactored.csv
This is the dataset that was annotated if an antithesis is present.

Please cite this work as follows:
```
@inproceedings{kuhn2024antithesis,
  title = {{Using Pre-trained Language Models in an End-to-End Pipeline for Antithesis Detection}},
  booktitle = {Proceedings of the 2024 Joint International Conference On Computational Linguistics, Language Resources And Evaluation},
  author = {K{\"u}hn, Ramona and Saadi, Khouloud and Mitrovi{\'c}, Jelena and Granitzer, Michael},
  year = {2024},
  month = may,
  address = {Torino, Italy},
}
```

## Acknowledgement
The project on which this report is based was funded by the German Federal Ministry of Education and Research (BMBF) under the funding code 01IS20049. The authors are responsible for the content of this publication.

<img src="https://github.com/user-attachments/assets/5e1ca975-704b-417b-958a-9fbfb6a893d8" width="400" height="300">


