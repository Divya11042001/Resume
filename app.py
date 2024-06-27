import streamlit as st
import pickle
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix


nltk.download('punkt')
nltk.download('stopwords')

#loading model
clf=pickle.load(open("clf.pkl","rb"))
tfidf=pickle.load(open("tfidf.pkl","rb"))

def clean_resume(resume_text):
    clean_text=re.sub("http\S+\s","",resume_text)
    clean_text = re.sub(r'[^a-zA-Z\s]','', clean_text)
    clean_text = clean_text.replace('\r\n', '')

    return clean_text



#web app
def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader("Upload_Resume",type=['txt','pdf'])
    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #if utf-8 decoding fails try to decode with 'latin-1'
            resume_text=resume_bytes.decode('latin-1')


        cleaned_resume=clean_resume(resume_text)
        input_features=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        st.write(prediction_id)

        #map category  Id to Name
        category_mapping={
            6 :'Data Science',
            12 :'HR',
            0 :'Advocate',
            1 :'Arts',
            24 :'Web Designing',
            16 :'Mechanical Engineer',
            22 : 'Sales',
            14 : 'Health and fitness',
            5 : 'Civil Engineer',
            15 :'Java Developer',
            4 :'Business Analyst',
            21 :'SAP Developer',
            2 :'Automation Testing',
            11 :'Electrical Engineering',
            18:'Operations Manager',
            20 :'Python Developer',
            8 :'DevOps Engineer',
            17 :'Network Security Engineer',
            19 :'PMO',
            7 :'Database',
            13: 'Hadoop',
            10 : 'ETL Developer',
            9 : 'DotNet Developer',
            3 : 'Blockchain',
            23:'Testing'
            }

        category_name=category_mapping.get(prediction_id,"Unknown")

        st.write("Prediction category",category_name)


