# vedaant-jain-sectioning-info-extraction

# Section-Based-IE-From-Academic-Resumes

## Overview

This module is responsible for sectioning an academic resume into sections according to titles, and then using a fine-tuned version of BERT to extract information from education, employment, and publication sections. 

## Setup

List the steps needed to install your module's dependencies: 

1. Python version: 3.9.6, cuda version: 11.1.

2. Install DiT and detectron2 using the instructions here: https://github.com/microsoft/unilm/tree/master/dit

3. Then run the requirements.txt file in the root folder in this repository. 

4. Include instructions on how to run any tests you have written to verify your module is working properly. 

It is very important to also include an overall breakdown of your repo's file structure. Let people know what is in each directory and where to look if they need something specific. This will also let users know how your repo needs to structured so that your module can work properly

```
vedaant-jain-sectioning-info-extraction/
    - requirements.txt
    - data/ 
        -- text_data/
            --- 
        -- title_annotated_images/
            --- result(1-4).json
            --- result(5- ).json
        -- resume_pdfs/
        --resume_images/
    - Inference/
        -- Education_Model.py
        -- Employment_Model.py
        -- Publication_Model.py
        -- ResumeInfoExtractor.py
        -- ResumeSeqTagger.py
        -- SectionDivider.py
        -- RunInference.ipynb
        -- RunSectionDivider.ipynb
    - Text-Based-IE/
        -- EducationIE.ipynb
        -- EmploymentIE.ipynb
        -- PublicationsIE.ipynb
    - title-detection-evaluation/
        -- DivideImage.ipynb

``` 
* `Text-Based-IE/`: contains notebooks to train the models for different section's information extraction
* `Inference/`: contains scripts for running inference
* `Inference/ResumeSeqTagger.py`: base class from which the education, employment, and publication model classes inherit methods
* `Inference/ResumeInfoExtractor`: this class has memebers for sectioning a resume and also extracting information from appropriate sections
* `Inference/RunInference.ipynb`: extracts information from a resume page image
* `Inference/RunSectionDivider.ipynb`: extracts section from a resume page image
* `title-detection-evaluation/DivideImage.ipynb`: evaluates performance of model to recognise titles based on bounding boxes from resumes pages using the data
* `data/text_data/`: contains data in CONLL format for sequence tagging for education, employment, publications
* `data/title_annotated_images/`: the result.json files contain annotations for the images of resume pages labelling the title bounding boxes
* `data/resume_images/`: contains images of resume pages corresponding to the resume pdfs present in the resume_pdfs folder, Note all resume images have not been included because of size but can be extracted from the corresponding pdfs.

## Functional Design (Usage)

* Takes as input a string text, representing the text in a section, returns the information in the string depending which model called it. 
```python, Education_Model.py
    def Get_InfoAll(text: str):
        ... 
        return {'Degree': ['PH. D. in Computer Science', ...],
    'University': ['University of Florida, Gville, USA' , ... ],
    'Thesis':  ['Designing and Leveraging Trustworthy Provenance - Aware Architectures', ...]}
```
Note for Publication Model, and Employment model, the output fields are: [Authors, Journal, Title] and [Designation, Employer] respectively.

* Takes as input an image(numpy array, RGB), representing the a section, Outputs the information present in the image of resume section 
```python, Education_Model.py
    def Get_Info(image: np.array):
        ... 
        return {'Degree': ['PH. D. in Computer Science', ...],
    'University': ['University of Florida, Gville, USA' , ... ],
    'Thesis':  ['Designing and Leveraging Trustworthy Provenance - Aware Architectures', ...]}
```
Note for Publication Model, and Employment model, the output fields are: [Authors, Journal, Title] and [Designation, Employer] respectively.

* Takes as input a sectiondivider object, checkpoints for education, employment, and publication model, a title map to separate headings and subheadings and their categories into education, employment, publication. The only required parameter is SectionDivider object, the rest are optional, and have default values. The model checkpoint's have the default value in the format "{}_bert_seq_tag" where either of education, employment, or publication replaces the {}. Another parameter is the CreateAllModels(default is False), unless passed as true, the model for information extraction will not be created and will have to be passed as arguments to methods that carry out the information extraction(see below).  
```python, ResumeInfoExtractor.py
    def __init__(self, section_divider, Create_All_Models = False, education_model_checkpoint = None, employment_model_checkpoint = None, publication_model_checkpoint = None, title_map = None) -> None:
```

* Takes as input an image(numpy array, RGB), representing the page of a resume, outputs the information in that image as a dictionary, inputs also include education, employment, and publication model if the user wishes to override the existing models present as members of the ResumeInfoExtractor instance.
```python, ResumeInfoExtractor.py
    def Extract_InfoAll(self, images, education_model = None, employment_model = None, publication_model = None):
        ... 
        return {'Education': [...],
    'Employment': [... ],
    'Publication':  [ ...]}
```
* Takes as input a list of image(numpy array, RGB), representing the pages of a resume, outputs the information in those pages as a dictionary, inputs also include education, employment, and publication model if the user wishes to override the existing models present as members of the ResumeInfoExtractor instance.
```python, ResumeInfoExtractor.py
    def Extract_Info(self, images, education_model = None, employment_model = None, publication_model = None):
```
## Algorithmic Design 
This section should contain a detailed description of all different components and models that you will be using to achieve your task as well as a diagram. Here is a very basic example of what you should include:

We generate vector representations for each document using BERT, we then train a simple, single-layer fully connected neural network using the documents and labels from the training set.

First, we select a set of labeled text documents `d_1, d_2, …, d_N` from the arxiv dataset available on Kaggle. The documents are randomly partitioned into two sets for training and testing. We use the BERT language model's output as the input to the neural network. Only the weights of the neural network are modified during training. 

After training, we run the trained model to classify the test documents into one of the classes in C. Below is a picture of the architecture of the module. The diagram below was constructed using draw.io 


![design architecture](https://github.com/Forward-UIUC-2021F/guidelines/blob/main/template_diagrams/sample-design.png)





## Issues and Future Work

In this section, please list all know issues, limitations, and possible areas for future improvement. For example:

* High false negative rate for document classier. 
* Over 10 min run time for one page text.
* Replace linear text search with a more efficient text indexing library (such as whoosh)
* Include an extra label of "no class" if all confidence scores low. 


## Change log

Use this section to list the _major_ changes made to the module if this is not the first iteration of the module. Include an entry for each semester and name of person working on the module. For example 

Fall 2021 (Student 1)
* Week of 04/11/2022: added two new functions responsible for ...
* Week of 03/14/2022: fixed bug and added support for ...

Spring 2021 (Student 2)
...

Fall 2020 (Student 3)
...


## References 
include links related to datasets and papers describing any of the methodologies models you used. E.g. 

* Dataset: https://www.kaggle.com/Cornell-University/arxiv 
* BERT paper: Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Include a brief summary of your module here. For example: this module is responsible for classifying pieces of text using a neural network on top of BERT. 

Note: if this is a second or latter iteration of a module, you may reuse the old iteration's README as a starting point (you should still update it). 
