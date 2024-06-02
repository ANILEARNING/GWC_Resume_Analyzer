import random
import re
import time
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import json
import google.generativeai as genai
import json

key = st.secrets["API_KEY"]

genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-pro')

project_and_curriculum = {
  "data_analyst": {
    "curriculum": {
      "Day 1": {
        "topic": "Introduction to Data Analysis",
        "subtopics": [
          "What is data analysis?",
          "Key concepts in data analysis",
          "Basic understanding of data and types of data"
        ]
      },
      "Day 2": {
        "topic": "Data Collection and Cleaning",
        "subtopics": [
          "Sources of data",
          "Data cleaning techniques",
          "Handling missing values"
        ]
      },
      "Day 3": {
        "topic": "Exploratory Data Analysis (EDA)",
        "subtopics": [
          "Descriptive statistics",
          "Data visualization techniques",
          "Identifying patterns and trends"
        ]
      },
      "Day 4": {
        "topic": "Data Visualization",
        "subtopics": [
          "Creating charts and graphs",
          "Using tools like Matplotlib and Seaborn",
          "Dashboard creation with Tableau/Power BI"
        ]
      },
      "Day 5": {
        "topic": "Basic SQL for Data Analysis",
        "subtopics": [
          "Introduction to SQL",
          "Basic queries and data manipulation",
          "Joining tables and data aggregation"
        ]
      }
    },
    "project_recommendations": [
      "Sales Data Analysis",
      "Customer Segmentation",
      "Market Basket Analysis",
      "Employee Performance Analysis",
      "Website Traffic Analysis"
    ]
  },
  "data_scientist": {
    "curriculum": {
      "Day 1": {
        "topic": "Introduction to Data Science",
        "subtopics": [
          "What is data science?",
          "Key concepts in data science",
          "Overview of data science lifecycle"
        ]
      },
      "Day 2": {
        "topic": "Statistics for Data Science",
        "subtopics": [
          "Probability and statistics fundamentals",
          "Hypothesis testing",
          "Descriptive and inferential statistics"
        ]
      },
      "Day 3": {
        "topic": "Data Preprocessing",
        "subtopics": [
          "Data cleaning techniques",
          "Feature engineering",
          "Scaling and normalization"
        ]
      },
      "Day 4": {
        "topic": "Introduction to Machine Learning",
        "subtopics": [
          "Supervised vs unsupervised learning",
          "Common machine learning algorithms",
          "Model evaluation and validation"
        ]
      },
      "Day 5": {
        "topic": "Advanced Machine Learning",
        "subtopics": [
          "Ensemble methods",
          "Deep learning basics",
          "Model deployment and monitoring"
        ]
      }
    },
    "project_recommendations": [
      "Customer Churn Prediction",
      "Sentiment Analysis of Product Reviews",
      "Image Classification",
      "Recommendation Systems",
      "Fraud Detection"
    ]
  },
  "data_engineer": {
    "curriculum": {
      "Day 1": {
        "topic": "Introduction to Data Engineering",
        "subtopics": [
          "What is data engineering?",
          "Key concepts in data engineering",
          "Overview of data pipelines"
        ]
      },
      "Day 2": {
        "topic": "Data Warehousing",
        "subtopics": [
          "Introduction to data warehousing",
          "ETL processes",
          "Data warehousing solutions (e.g., Redshift, BigQuery)"
        ]
      },
      "Day 3": {
        "topic": "Big Data Technologies",
        "subtopics": [
          "Introduction to Hadoop and Spark",
          "Big data processing frameworks",
          "Real-time data processing"
        ]
      },
      "Day 4": {
        "topic": "Database Management",
        "subtopics": [
          "Relational vs NoSQL databases",
          "Database design and modeling",
          "Database performance tuning"
        ]
      },
      "Day 5": {
        "topic": "Data Pipeline Automation",
        "subtopics": [
          "Building data pipelines with Apache Airflow",
          "Automating ETL processes",
          "Monitoring and maintaining data pipelines"
        ]
      }
    },
    "project_recommendations": [
      "ETL Pipeline for E-commerce Data",
      "Real-time Analytics Dashboard",
      "Data Lake Implementation",
      "Stream Processing with Kafka",
      "Data Warehouse Optimization"
    ]
  },
  "python_dev": {
    "curriculum": {
      "Day 1": {
        "topic": "Introduction to Python",
        "subtopics": [
          "Setting up Python environment",
          "Basic syntax and data types",
          "Control structures (if, for, while)"
        ]
      },
      "Day 2": {
        "topic": "Data Structures and Algorithms",
        "subtopics": [
          "Lists, tuples, and dictionaries",
          "Basic algorithms (searching, sorting)",
          "Complexity analysis"
        ]
      },
      "Day 3": {
        "topic": "Object-Oriented Programming",
        "subtopics": [
          "Classes and objects",
          "Inheritance and polymorphism",
          "Encapsulation and abstraction"
        ]
      },
      "Day 4": {
        "topic": "Python Libraries and Frameworks",
        "subtopics": [
          "Introduction to popular libraries (e.g., NumPy, Pandas)",
          "Web development with Django/Flask",
          "Working with APIs"
        ]
      },
      "Day 5": {
        "topic": "Testing and Debugging",
        "subtopics": [
          "Writing unit tests",
          "Using debugging tools",
          "Best practices for debugging and testing"
        ]
      }
    },
    "project_recommendations": [
      "Web Scraping Tool",
      "REST API with Flask",
      "Data Analysis with Pandas",
      "Automated Testing Suite",
      "Personal Finance Tracker"
    ]
  },
  "full_stack_dev": {
    "curriculum": {
      "Day 1": {
        "topic": "Introduction to Full Stack Development",
        "subtopics": [
          "Overview of front-end and back-end",
          "Setting up development environment",
          "Understanding MVC architecture"
        ]
      },
      "Day 2": {
        "topic": "Front-End Development",
        "subtopics": [
          "HTML, CSS, and JavaScript basics",
          "Introduction to front-end frameworks (e.g., React, Angular)",
          "Responsive web design"
        ]
      },
      "Day 3": {
        "topic": "Back-End Development",
        "subtopics": [
          "Server-side programming with Node.js/Python",
          "Database integration (SQL/NoSQL)",
          "Building RESTful APIs"
        ]
      },
      "Day 4": {
        "topic": "Full Stack Development Tools",
        "subtopics": [
          "Version control with Git",
          "Containerization with Docker",
          "CI/CD pipelines"
        ]
      },
      "Day 5": {
        "topic": "Deploying and Scaling Applications",
        "subtopics": [
          "Deploying applications on cloud platforms",
          "Scalability and performance optimization",
          "Monitoring and maintaining applications"
        ]
      }
    },
    "project_recommendations": [
      "E-commerce Website",
      "Social Media Platform",
      "Project Management Tool",
      "Real-time Chat Application",
      "Personal Portfolio Website"
    ]
  },
  "conclusion": "Prioritize daily learning with consistency as the key. Focus on recommended skills, projects, and curriculum. Showcase your work to strengthen your resume and demonstrate continuous growth and proficiency."
}

# Function to render curriculum based on selected education level
def render_curriculum(education_level):
    curriculum = project_and_curriculum.get(education_level, {}).get('curriculum', {})
    if curriculum:
        st.subheader(f"Personalized Curriculum for {education_level.capitalize()}")
        for day, day_data in curriculum.items():
          expander_title = f"{day[4:]}: {day_data['topic']}"
          with st.expander(expander_title):
            for subtopic in day_data['subtopics']:
                st.write(f"- {subtopic}")
    else:
        st.write("No curriculum found for selected education level.")

def show_personalized_curriculum_llm(data):
    st.subheader("Personalized Curriculum")
    for day, day_data in data.items():
        expander_title = f"{day[4:]}: {day_data['topic']}"
        with st.expander(expander_title):
            for subtopic in day_data['subtopics']:
                st.write(f"- {subtopic}")

# Function to render project recommendations based on selected education level
def render_project_recommendations(education_level):
    recommendations = project_and_curriculum.get(education_level, {}).get('project_recommendations', [])
    if recommendations:
        st.subheader("Project Recommendations:")
        for recommendation in recommendations:
            st.write(f"- {recommendation}")
    else:
        st.sidebar.write("No project recommendations found for selected education level.")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to calculate similarity between texts
def calculate_similarity(user_resume_text, job_description):
    corpus = [user_resume_text, job_description]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1] * 100
    return similarity_score

def extract_score(score_str):
    # Regular expression to match numbers
    number_pattern = r'\d+'

    # Regular expression to match percentage format
    percentage_pattern = r'(\d+)%'

    # Regular expression to match "out of" format
    out_of_pattern = r'(\d+) out of (\d+)'

    # Regular expression to match number followed by percentage format
    number_percentage_pattern = r'(\d+)%'

    # Regular expression to match percentage followed by "out of" format
    percentage_out_of_pattern = r'(\d+)% out of (\d+)'

    # Regular expression to match number followed by percentage followed by "out of" format
    number_percentage_out_of_pattern = r'(\d+)% out of (\d+)'

    # Try to match each pattern to the input string
    match_number = re.match(number_pattern, score_str)
    match_percentage = re.match(percentage_pattern, score_str)
    match_out_of = re.match(out_of_pattern, score_str)
    match_number_percentage = re.match(number_percentage_pattern, score_str)
    match_percentage_out_of = re.match(percentage_out_of_pattern, score_str)
    match_number_percentage_out_of = re.match(number_percentage_out_of_pattern, score_str)

    # Extract the score based on the matched pattern
    if match_number:
        return int(match_number.group(0))
    elif match_percentage:
        return int(match_percentage.group(1))
    elif match_out_of:
        return int(match_out_of.group(1))
    elif match_number_percentage:
        return int(match_number_percentage.group(1))
    elif match_percentage_out_of:
        return int(match_percentage_out_of.group(1))
    elif match_number_percentage_out_of:
        return int(match_number_percentage_out_of.group(1))
    else:
        return None  # If no match is found, return None

def extract_phone_numbers(text):
    phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    return phone_numbers

def extract_email(text):
    # Regular expression pattern for matching email addresses
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Find all matches of email addresses in the text
    emails = re.findall(pattern, text)
    return emails

def get_missing_skills(data):
    recommended_skills = [
        "Exploratory data analysis", "Data visualization",
        "Regression analysis", "Classification", "Cluster analysis", "Neural networks", "Natural language processing",
        "Python", "R", "SQL",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "NLTK", "TensorFlow", "Keras",
        "PowerBI", "Tableau", "Excel", "IBM SPSS",
        "Leadership", "Communication (written and verbal)", "Decision-making", "Quantitative analysis", "Presentation skills"
    ]
    missing_skills = [skill for skill in recommended_skills if skill not in data['skills']]
    return missing_skills

def get_random_quote():
    quotes = [
        "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
        "The harder you work, the luckier you get. - Gary Player",
        "The only way to do great work is to love what you do. - Steve Jobs",
        "In cricket, as in life, loyalty is everything. - Rahul Dravid",
        "I have failed at times, but I never stopped trying. - Sachin Tendulkar",
        "Cricket is a team game. If you want fame for yourself, go play an individual game. - Gautam Gambhir",
        "It's about not giving up. Failures are a part of life. If you don't fail, you don't learn. If you don't learn, you'll never change. - Cheteshwar Pujara",
        "Self-belief and hard work will always earn you success. - Virat Kholi",
        "Face the failure, until the failure fails to face you. - MS Dhoni",
    ]
    return random.choice(quotes)
# Call the function to get missing skills

# Function to display JSON data
def display_json_data(data):
    st.header("ATS Report and Feedback:")
    st.title("Resume Information")
    # st.success(f"**Hello {data['name']}, Happy to have you here...Lets Explore the Analysis Resport **")
    st.success(f"**Hey there, {data['name']}! Welcome aboard! Let's dive into your analysis report and uncover some insights together!**")
    st.subheader("**Resume Score📝**")
    # st.progress(int(data['score']))
    # progress_bar = st.progress(0,)
    # score_int = round(int(data['score']),2)
    # for percent_complete in range(score_int + 1):
    #     progress_bar.progress(percent_complete)
    #     time.sleep(0.05)
    

    progress_bar = st.progress(0,)
    score = data['score']
    if isinstance(score, (int, float)):
        score_int = round(score, 2)
        for percent_complete in range(score_int + 1):
            progress_bar.progress(percent_complete)
            time.sleep(0.05)
        if score_int >= 70:
          st.balloons()
    elif isinstance(score, str):
        st.warning(f"Your score is '{score}'")

    st.write(f"Your Resume Score is: {score_int}%")

    st.subheader("Summary: ")
    with st.expander("Breif summary based on your education, certification, skills and experience💡"):
        st.write(data['summary'])
    if not data['personalized_curriculum']:
        education = data['highest_education']
        if len(education) > 0:
            render_curriculum(education)
    else:
        show_personalized_curriculum_llm(data['personalized_curriculum'])

    # Display Skills
    st.subheader("Skills")
    skills_html = ""
    for skill in data['skills']:
        skills_html += f"<div style='background-color:#3498db;color:white;padding:8px;border-radius:5px;margin-right:5px;display:inline-block;'>{skill}</div>"
    st.write(skills_html, unsafe_allow_html=True)
    
    
    # Display Recommended Skills
    if not data['suggested_skills']:
      missing_skills = get_missing_skills(data)
      st.subheader("Recommended Skills to Upskill")
      recommended_skills_html = ""
      for recommended_skill in missing_skills:
        recommended_skills_html += f"<div style='background-color:#2ecc71;color:white;padding:12px;border-radius:12px;margin-right:12px;display:inline-block;'>{recommended_skill}</div>"
      st.write(recommended_skills_html, unsafe_allow_html=True)       
      st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost the chances of getting a Job</h4>''', unsafe_allow_html=True)
      st.write("")
    else:
      st.subheader("Recommended Skills to Upskill")
      recommended_skills_html = ""
      for recommended_skill in data['suggested_skills']:
        recommended_skills_html += f"<div style='background-color:#2ecc71;color:white;padding:8px;border-radius:10px;margin-right:8px;display:inline-block;'>{recommended_skill}</div>"
      st.write(recommended_skills_html, unsafe_allow_html=True)       
      st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost the chances of getting a Job</h4>''', unsafe_allow_html=True)
      st.write("")

    # Display feedback
    st.subheader("Feedback")
    with st.expander("Positive Points"):
        for point in data['feedback']['positive_points']:
            st.write(f"- {point}")
    with st.expander("Negative Points"):
        for point in data['feedback']['negative_points']:
            st.write(f"- {point}")
    with st.expander("Suggestions"):
        for suggestion in data['feedback']['suggestions']:
            st.write(f"- {suggestion}")
    
    if not data['project_recommendations']: 
        education = data['highest_education']
        if len(education) > 0:
             render_project_recommendations(education)
    else:
        st.subheader("Project Recommendations")
        with st.expander(f"{data['job_title']} Projects"):
            for project in data['project_recommendations']:
                  st.write(f"     - {project}")
    try:
      if data['conclusion'] is not None:
        st.subheader('Conclusion')
        st.write(data['conclusion'])
      else:
          st.subheader('Conclusion')
          st.write("Hi " + data["name"] +" " + project_and_curriculum["conclusion"])
    except:
        pass
        

# Function to generate JSON data based on user input
def generate_json_data(user_pdf_text, job_requirement_text):
    emails = extract_email(user_pdf_text)
    mobiles = extract_phone_numbers(user_pdf_text)
    for email in emails:
        sec_user_pdf_text = user_pdf_text.replace(email,"")
    for mobile in mobiles:
        sec_user_pdf_text = user_pdf_text.replace(mobile,"")
    
    input_prompt = f'''
    You are an ATS for Data Analyst, Data Engineer, Data Scientist, Full Stack Developer and Python Developer Resumes. I will give you Text from the PDF resumes, and you have to give me:
    - name
    - score out of 100 compared with Job Description (should be a number)
    - skills
    - suggested_skills for their role (not mentioned in the resume)
    - job_title
    - project_recommendations
    - feedback
        - positive_points
        - negative_points
        - suggestions
    - experience
        - title
        - company
        - location
        - start_date
        - end_date
    - age
    - highest_education (Output_classes : school, bachelors, masters)
    - personalized curriculum example_format:{project_and_curriculum["data_analyst"]}
        - day (eg. day_1,day_2,day_3,day_4,day_5)
        - topic 
        - subtopics
    - certifications
    - breif summary based on education, certification, skills and experience 
    - Conclusion 
    Please try your best when giving suggestions and recommendations, Language is strictly english. 
    Only give response in JSON so I can parse it using Python and use it later!
    
    Give a Response based on Role of the resume it must have the following keys (if not found, return values as empty string consider all the fields are required)
    - name
    - score
    - skills
    - age
    - highest_education
    - suggested_skills
    - job_title
    - project_recommendations
    - feedback
        - positive_points
        - negative_points
        - suggestions
    - summary
    - personalized_curriculum
        - day (eg. day1,day2,day3)
        - topic 
        - subtopics
    - conclusion 

    Here is the PDF Text: {sec_user_pdf_text} and Job Description Text:{job_requirement_text}. (consider all the fields are required)
    '''
    response = model.generate_content(input_prompt)
    response_json = response.text.replace('```', '')
    response_json = response_json.replace('json', '', 1)
    response_json = response_json.replace('JSON', '', 1)
    data = json.loads(response_json)

    if isinstance(data['score'], str):
      data['score'] = extract_score(data['score'])
    if data['score'] is None or int(data['score'])<=30 :
        data['score'] = calculate_similarity(user_pdf_text, job_requirement_text)
    if int(data['score'])<=40:
        data['score'] = random.randint(40, 50)

    print(data)
    return data

# Main function for Streamlit app
def main():
      st.title(""":violet[GWC Data.AI] - :orange[Resume Analyzer]🪄""")
      
      st.markdown("""
            ### Welcome to My AI Project

            This project is crafted to **streamline the interview process** by swiftly evaluating candidates. I, **Anish S**, developed this project to demonstrate my expertise and skills in **data science and AI**.

            ---

            #### Analyze candidates Resume Against Any Job Description

            If you want to analyze your resume against any job description, this tool will provide you with **valuable insights** to better match your skills with the job requirements.

            ---

            *Feel free to upload your resume and job description below to get started!*
            """)
      uploaded_file = st.file_uploader("**Upload your Resume/CV (PDF)**", type="pdf")
      if uploaded_file is not None:
          pdf_contents = uploaded_file.read()
          user_pdf_text = extract_text_from_pdf(uploaded_file)
        #   user_pdf_base64 = base64.b64encode(pdf_contents).decode('utf-8')
        #   pdf_display = f'''
        #       <iframe 
        #           src="data:application/pdf;base64,{user_pdf_base64}" 
        #           width="700" 
        #           height="500" 
        #           style="border: none;"
        #       ></iframe>
        #   '''
          
          # Use st.components.v1.html to render the HTML
        #   st.components.v1.html(pdf_display, height=500, width=700)
          st.write("")
          with st.expander("**🔍 Insert here...If you want to analyze your resume with any Job Description**"):
            st.write("<h4>Select input method for Job Description:</h4>", unsafe_allow_html=True)
            input_method = st.radio("",
                                ("Text",
                                "PDF"))
            st.write("**Note:** Select 'Text' if you have JD in text format | Select 'PDF' if you have JD in PDF format or if you want to compare your resume with another resume.")
            jd_input = None
            st.write("")
          
            if input_method == "Text":
                jd_input = st.text_area("**Paste or type the job description here:**", height=200)
                # if st.button("Analyze", key="analyze_button_text"):
                
            elif input_method == "PDF":
                comparing_file = st.file_uploader("**Upload the CV you want to compare (PDF)**", type="pdf")
                if comparing_file is not None:
                    jd_input = extract_text_from_pdf(comparing_file)
                    # analyze_button_pdf = st.button("""Analyze""",
                    #                               key="analyze_button_pdf", 
                    #                               help="Click to analyze the PDF"
                    #                               )
                    # if analyze_button_pdf:
                    # # if st.button("Analyze PDF", key="analyze_button_pdf"):
                    #     with st.spinner('Loading...' + f"{get_random_quote()}"):
                    #         data = generate_json_data(user_pdf_text, jd_input)
                    #         display_json_data(data)

          analyze_button_text = st.button("Analyze""",
                                                  key="analyze_button_text", 
                                                  help="Click to analyze your Resume"                                               
                                                  )
          # if analyze_button_text:
          #           # with st.spinner('Analyzing job description...'):
          #           with st.spinner("Loading..." + f"**{get_random_quote()}**"):
          #               data = generate_json_data(user_pdf_text, jd_input)
          #           display_json_data(data)
          if analyze_button_text:
                    if input_method == "Text":
                        with st.spinner("Loading..." + f"**{get_random_quote()}**"):
                            data = generate_json_data(user_pdf_text, jd_input)
                        display_json_data(data)
                    elif input_method == "PDF":
                        with st.spinner("Loading..." + f"**{get_random_quote()}**"):
                          data = generate_json_data(user_pdf_text, jd_input)
                        display_json_data(data)
                    else:
                        with st.spinner("Loading..." + f"**{get_random_quote()}**"):
                          data = generate_json_data(user_pdf_text, jd_input)
                        display_json_data(data)
    # else:
    #   st.title("Please enter email and password to access the content")

# Run the Streamlit app
if __name__ == "__main__":
    main()
