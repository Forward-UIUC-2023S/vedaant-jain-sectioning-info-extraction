from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import DataCollatorForTokenClassification
import easyocr

from ResumeSeqTagger import ResumeSeqTagger

class EmploymentInformationExtractor(ResumeSeqTagger):
    def __init__(self, model_checkpoint='bert_seq_tag', ocr_reader = None) -> None:
        super().__init__(model_checkpoint, ocr_reader)

        self.designation_idx = (self.label2id['B-Designation'], self.label2id['I-Designation'])
        self.employer_idx = (self.label2id['B-Employer'], self.label2id['I-Employer'])
    
    def Clean_Designation(self, preds):
        return self.Clean_Section(preds, self.designation_idx[0], self.designation_idx[1])
    
    def Clean_Employer(self, preds):
        return self.Clean_Section(preds, self.employer_idx[0], self.employer_idx[1])
    
    def Clean_Preds(self, preds):
        preds = self.Clean_Designation(preds)
        preds = self.Clean_Employer(preds)
        return preds
    
    def Get_InfoAll(self, text):
        batch = self.Get_Batch(text)
        preds = self.GetPreds(batch)
        information_extracted = {'Designation': self.Get_SecInfo(batch, preds, self.designation_idx[0], self.designation_idx[1]),
                                 'Employer': self.Get_SecInfo(batch, preds, self.employer_idx[0], self.employer_idx[1])}
        return information_extracted

    def Extract_Info(self, image):
        text = self.Get_Text(image)
        return self.Get_InfoAll(text)
