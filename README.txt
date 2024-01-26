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