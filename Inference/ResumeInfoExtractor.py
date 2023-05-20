from Education_Model import EducationInformationExtractor
from Employment_Model import EmploymentInformationExtractor
from Publication_Model import PublicationsInformationExtractor

class ResumeInformationExtractor():
    def __init__(self, section_divider, Create_All_Models = False, education_model_checkpoint = None, employment_model_checkpoint = None, publication_model_checkpoint = None, title_map = None) -> None:
        self.section_divider = section_divider
        if Create_All_Models:
            if education_model_checkpoint is None:
                self.education_model = EducationInformationExtractor('education_bert_seq_tag', self.section_divider.reader)
            else:
                self.education_model = EducationInformationExtractor(education_model_checkpoint, self.section_divider.reader)
            if employment_model_checkpoint is None:
                self.employment_model = EmploymentInformationExtractor('employment_bert_seq_tag', self.section_divider.reader)
            else:
                self.employment_model = EmploymentInformationExtractor(employment_model_checkpoint, self.section_divider.reader)
            
            if publication_model_checkpoint is None:
                self.publication_model = PublicationsInformationExtractor('publication_bert_seq_tag', self.section_divider.reader)
            else:
                self.publication_model = PublicationsInformationExtractor(publication_model_checkpoint, self.section_divider.reader)
        else:
            self.education_model = None
            self.employment_model = None
            self.publication_model = None

        if title_map is None:
            temp = {}
            temp['Education'] = ['EDUCATION']
            temp['Employment'] = ['EMPLOYMENT', 'PROFESSIONAL EXPERIENCE', 'WORK EXPERIENCE', 'INDUSTRIAL EXPERIENCE', 'INDUSTRY EXPERIENCE', 'EXPERIENCE', 'ACADEMIC APPOINTMENTS', 'PROFESSIONAL EMPLOYMENT']
            temp['Publications'] = ['PUBLICATIONS', 'Journal Publications', 'Conference Publications', 'Working Papers', 'OTHER PUBLICATIONS', 'CONFERENCE PARTICIPATION','COMPLETED RESEARCH ARTICLES', 'Working Papers', 'Publications', '']
            self.titles_map = {}
            for key in temp:
                for title in temp[key]:
                    title = title.lower()
                    title_str = title.replace(' ', '')
                    self.titles_map[title_str] = key
        else:
            self.titles_map = title_map
        
    def Get_Sections(self, image):
        return self.section_divider.GetSections(image)
    
    def Get_Education(self, image, model = None):
        if self.education_model is None and model is not None:
            self.education_model = model
        if self.education_model is None:
            print("No Education Model initialized or passed as argument")
            return None
        return self.education_model.Extract_Info(image)
    
    def Get_Employment(self, image, model = None):
        if self.employment_model is None and model is not None:
            self.employment_model = model
        if self.employment_model is None:
            print("No Employment Model initialized or passed as argument")
            return None
        return self.employment_model.Extract_Info(image)
    
    def Get_Publications(self, image, model = None):
        if self.publication_model is None and model is not None:
            self.publication_model = model
        if self.publication_model is None:
            print("No Publications Model initialized or passed as argument")
            return None
        return self.publication_model.Extract_Info(image)
    
    def Extract_Info(self, image, last_title = None, education_model = None, employment_model = None, publication_model = None):
        sections, texts = self.Get_Sections(image)
        all_info = {}
        all_info['Education'] = []
        all_info['Employment'] = []
        all_info['Publications'] = []
        image = sections[0]
        if last_title is not None:
            if last_title == 'Education':
                all_info['Education'].append(self.Get_Education(image, education_model))
            elif last_title == 'Employment':
                all_info['Employment'].append(self.Get_Employment(image, employment_model))
            elif last_title == 'Publications':
                all_info['Publications'].append(self.Get_Publications(image, publication_model))
        
        for i, text in enumerate(texts):
            text = text[1]
            text = text.lower()
            text = text.replace(' ', '')
            if text in self.titles_map:
                title = self.titles_map[text]
                image = sections[i+1]
                if title == 'Education':
                    all_info['Education'].append(self.Get_Education(image, education_model))
                elif title == 'Employment':
                    all_info['Employment'].append(self.Get_Employment(image, employment_model))
                elif title == 'Publications':
                    all_info['Publications'].append(self.Get_Publications(image, publication_model))
                last_title = title
        return all_info, last_title
    
    def Extract_InfoAll(self, images, education_model = None, employment_model = None, publication_model = None):
        all_info = {}
        all_info['Education'] = []
        all_info['Employment'] = []
        all_info['Publications'] = []
        last_title = None
        for image in images:
            info, last_title = self.Extract_Info(image, last_title, education_model, employment_model, publication_model)
            all_info['Education'].append(info['Education'])
            all_info['Employment'].append(info['Employment'])
            all_info['Publications'].append(info['Publications'])
            last_title = last_title
        return all_info