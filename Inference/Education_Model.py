from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import DataCollatorForTokenClassification
import easyocr

from ResumeSeqTagger import ResumeSeqTagger

class EducationInformationExtractor(ResumeSeqTagger):
    def __init__(self, model_checkpoint='bert_seq_tag', ocr_reader = None) -> None:
        super().__init__(model_checkpoint, ocr_reader)

        self.degree_idx = (self.label2id['B-Degree'], self.label2id['I-Degree'])
        self.institution_idx = (self.label2id['B-University'], self.label2id['I-University'])
        self.thesis_idx = (self.label2id['B-Thesis'], self.label2id['I-Thesis'])
    
    def Clean_Degree(self, preds):
        return self.Clean_Section(preds, self.degree_idx[0], self.degree_idx[1])
    
    def Clean_Institution(self, preds):
        return self.Clean_Section(preds, self.institution_idx[0], self.institution_idx[1])
    
    def Clean_Thesis(self, preds):
        return self.Clean_Section(preds, self.thesis_idx[0], self.thesis_idx[1])
    
    def Clean_Preds(self, preds):
        preds = self.Clean_Degree(preds)
        preds = self.Clean_Institution(preds)
        preds = self.Clean_Thesis(preds)
        return preds
    
    def Get_InfoAll(self, text):
        batch = self.Get_Batch(text)
        preds = self.GetPreds(batch)
        information_extracted = {'Degree': self.Get_SecInfo(batch, preds, self.degree_idx[0], self.degree_idx[1]),
                                 'University': self.Get_SecInfo(batch, preds, self.institution_idx[0], self.institution_idx[1]),
                                 'Thesis': self.Get_SecInfo(batch, preds, self.thesis_idx[0], self.thesis_idx[1])}
        return information_extracted

    def Extract_Info(self, image):
        text = self.Get_Text(image)
        return self.Get_InfoAll(text)