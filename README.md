# vedaant-jain-sectioning-info-extraction

# Section-Based-IE-From-Academic-Resumes

## Overview

This module is responsible for sectioning an academic resume into sections according to titles, and then using a fine-tuned version of BERT to extract information from education, employment, and publication sections. Project Report: [Click here](ResearchReportSpring2023.pdf)

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
    def Extract_InfoAll(self, image, last_title=None, education_model = None, employment_model = None, publication_model = None):
        ... 
        return {'Education': [...],
    'Employment': [... ],
    'Publication':  [ ...]}
```
* Takes as input a list of images(numpy array, RGB), representing the pages of a resume, outputs the information in those pages as a dictionary, inputs also include education, employment, and publication model if the user wishes to override the existing models present as members of the ResumeInfoExtractor instance.
```python, ResumeInfoExtractor.py
    def Extract_Info(self, images, education_model = None, employment_model = None, publication_model = None):
```

* Takes as input a word2vec model, iou_threshold for creating sections, config_file(example present in the repository under the Inference directory) name for configuring the detectron2 base constructor for SectionDivider class.
```python, SectionDivider.py
    def __init__(self, config_file = "config.yml", titles="default", iou_threshold=0.3, w2v_model = None):
        ... 
        return None
```

* Takes as input an image(numpy array, RGB), representing the page of a resume, outputs the sections of the image as a list of np.array and their titles as a list of strings. 
```python, SectionDivider.py
    GetSections(image)
        ... 
        return (Sections, texts)
```

## Video Link:
https://drive.google.com/file/d/1ouqA8Q1u7c4lR6wS3wNqmM66S_FN9BY4/view?usp=share_link

## Algorithmic Design 
- First we section a resume into sections based on titles.
 -- To do this, we use the DiT module from Microsoft to recognise the titles in images of resume pages, then as almost all academic resumes are vertically aligned, we divide the resume image into different sections. This is done by the SectionDivider Module.
 -- DiT also recognises sub-headings as headings. We differentiate between them using a set of common headings that academica resumes have. We compare the text of the headings and sub-headings to the set of common-headings using the Word2Vec embeddings. We can do this because academic resumes follow a common set of headings and sub-headings.
- After sectioning the resume into respective sections, we extract text using the easyOCR module.
- Information is then extracted from the text using models based on BERT fine-tuned on a small dataset to extract information like degree, university, designation.
- The model for information extraction is based on sequence tagging approach using the BIO tag.





## Issues and Future Work

In this section, please list all know issues, limitations, and possible areas for future improvement. For example:

* Lack of a good method to distinguish between headings and sub-headings 
* Using too many models at once especially for IE for each section.
* OCR-corrector based on the text of the pdf.


## References 
include links related to datasets and papers describing any of the methodologies models you used. E.g. 

[1] Cheng, Z., Zhang, P., Li, C., Liang, Q., Xu, Y., Li, P., Pu, S., Niu, Y., Wu,
F.: TRIE++: Towards End-to-End Information Extraction from Visually Rich
Documents (2022)
[2] Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang,
C., Che, W., Zhang, M., Zhou, L.: LayoutLMv2: Multi-modal Pre-training for
Visually-Rich Document Understanding (2022)
[3] Li, J., Xu, Y., Lv, T., Cui, L., Zhang, C., Wei, F.: DiT: Self-supervised Pre-training
for Document Image Transformer (2022)
[4] Alzaidy, R., Caragea, C., Giles, C.L.: Bi-lstm-crf sequence labeling for keyphrase
extraction from scholarly documents. In: The World Wide Web Conference. WWW
’19, pp. 2551–2557. Association for Computing Machinery, New York, NY, USA
(2019). https://doi.org/10.1145/3308558.3313642 . https://doi-org.proxy2.library.
illinois.edu/10.1145/3308558.3313642
[5] He, H., Choi, J.D.: Establishing Strong Baselines for the New Decade: Sequence
Tagging, Syntactic and Semantic Parsing with BERT (2020)
