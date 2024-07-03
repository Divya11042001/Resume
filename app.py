import streamlit as st
import pickle
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix
from streamlit_option_menu import option_menu
from PIL import Image


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

#set page icon
img=Image.open("images.png")
st.set_page_config(page_title="Resume_Screening_app",page_icon="img")
def about():
    st.title("Resume-Screening-App")
    st.subheader("1. Efficiency and Time-Saving:")
    st.write("Automating the initial review process significantly reduces the time spent on manual resume reviews, allowing hiring managers and recruiters to focus on interviewing and selecting the best candidates.")

    st.subheader("2. Enhanced Accuracy and Consistency:")
    st.write("Utilizing advanced algorithms and machine learning, the app ensures a consistent and unbiased evaluation of resumes, minimizing human errors and subjectivity.")

    st.subheader("3. Cost-Effectiveness:")
    st.write("By streamlining the resume screening process, companies can lower their overall hiring costs, including administrative expenses and the time spent by HR personnel.")

    st.subheader("4. Improved Candidate Experience:")
    st.write("Candidates receive quicker feedback on their applications, leading to a better overall experience and a positive perception of the company.")
    
    
    st.image("images.png",width=300)   


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

    

with st.sidebar:
    selected=option_menu(menu_title="Navigation",
                         options=["About","App Deployment"],
                         icons=["house","book"],
                         menu_icon="cast",
                         default_index=0)

if selected == 'About':
    about()
elif selected == 'App Deployment':
    main()


