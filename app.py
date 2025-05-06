from flask import Flask, request, send_file, jsonify, send_from_directory
import pandas as pd
import os
import PyPDF2
import docx
import textract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import Counter
from itertools import combinations
import re
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from threading import Lock


app = Flask(__name__, static_url_path='/static')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize lock for thread-safe operations
plot_lock = Lock()

# Load jobs.csv safely
try:
    jobs_csv_path = os.path.join(BASE_DIR, 'jobs.csv')
    df = pd.read_csv(jobs_csv_path, encoding='utf-8')
    print("✅ Loaded jobs.csv:", df.shape)
except Exception as e:
    print("❌ Error loading jobs.csv:", e)
    df = pd.DataFrame()

# Generate skills visualizations
def generate_skill_visualizations(df):
    try:
        static_dir = os.path.join(BASE_DIR, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        all_skills = []
        for skill_string in df['Skills Required'].dropna():
            skills = [s.strip().lower() for s in skill_string.split(',') if s.strip()]
            all_skills.append(skills)

        pair_counts = Counter()
        skill_set = set()

        for skills in all_skills:
            skill_set.update(skills)
            for combo in combinations(sorted(set(skills)), 2):
                pair_counts[combo] += 1

        skill_list = sorted(skill_set)
        matrix = pd.DataFrame(0, index=skill_list, columns=skill_list)

        for (skill1, skill2), count in pair_counts.items():
            matrix.at[skill1, skill2] = count
            matrix.at[skill2, skill1] = count

        matrix_path = os.path.join(static_dir, "skill_cooccurrence_matrix.csv")
        matrix.to_csv(matrix_path)
        print("✅ Saved:", matrix_path)

        skill_counts = Counter(skill for skills in all_skills for skill in skills)
        top_skills = [s for s, _ in skill_counts.most_common(20)]
        filtered = matrix.loc[top_skills, top_skills]

        nodes = [{'id': skill} for skill in filtered.columns]
        links = []

        for i, skill1 in enumerate(filtered.columns):
            for j, skill2 in enumerate(filtered.columns):
                if i < j and filtered.at[skill1, skill2] > 0:
                    links.append({
                        'source': skill1,
                        'target': skill2,
                        'value': int(filtered.at[skill1, skill2])
                    })

        network_path = os.path.join(static_dir, "skills_network.json")
        with open(network_path, "w") as f:
            json.dump({'nodes': nodes, 'links': links}, f, indent=2)
        print("✅ Saved:", network_path)
    except Exception as e:
        print("❌ Skill visualization generation error:", e)

if not df.empty:
    generate_skill_visualizations(df)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        text = ''
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ''

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ''

def extract_text_from_doc(file_path):
    try:
        return textract.process(file_path).decode('utf-8')
    except Exception as e:
        print(f"DOC extraction error: {e}")
        return ''

@app.route('/')
def home():
    return send_file(os.path.join(BASE_DIR, 'templates', 'index2.html'))

@app.route('/upload', methods=['GET'])
def upload_page():
    return send_file(os.path.join(BASE_DIR, 'templates', 'index.html'))

@app.route('/upload', methods=['POST'])
def upload_cv():
    if 'cv' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    uploaded_file = request.files['cv']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(uploaded_file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = uploaded_file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(file_path)

    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        cv_text = extract_text_from_pdf(file_path)
    elif ext == 'docx':
        cv_text = extract_text_from_docx(file_path)
    elif ext == 'doc':
        cv_text = extract_text_from_doc(file_path)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        os.remove(file_path)  # Clean up uploaded file
    except:
        pass

    if not cv_text.strip():
        return jsonify({'error': 'Could not extract text from CV'}), 400

    try:
        text_columns = ['Skills Required', 'Company Name', 'Profile Name', 'Experience', 'Salary', 'City']
        df['combined'] = df[text_columns].fillna('').agg(' '.join, axis=1)

        corpus = df['combined'].tolist() + [cv_text]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        df['score'] = cosine_similarities
        top_matches = df[df['score'] > 0].sort_values(by='score', ascending=False)
        top_matches = top_matches.drop_duplicates(subset=['URL', 'Profile Name'], keep='first').head(10)
        top_matches['percentage'] = (top_matches['score'] * 100).round(2)

        results = top_matches[[
            'Profile Name', 'Company Name', 'Skills Required', 'URL', 'Salary', 'City', 'percentage'
        ]].fillna('').to_dict(orient='records')

        matches_path = os.path.join(BASE_DIR, 'static', 'matches.json')
        with open(matches_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Error processing CV: {str(e)}'}), 500

@app.route('/results', methods=['GET'])
def get_results():
    try:
        matches_path = os.path.join(BASE_DIR, 'static', 'matches.json')
        with open(matches_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/job_titles', methods=['GET'])
def get_job_titles():
    try:
        job_titles = df['Profile Name'].dropna().unique().tolist()
        return jsonify(job_titles)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/company-trends', methods=['GET'])
def company_trends_page():
    return send_file(os.path.join(BASE_DIR, 'templates', 'company_trends.html'))

@app.route('/api/top-companies', methods=['GET'])
def top_companies_data():
    try:
        top_counts = Counter(df['Company Name'].dropna()).most_common(5)
        labels = [item[0] for item in top_counts]
        counts = [item[1] for item in top_counts]
        return jsonify({'labels': labels, 'counts': counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/skills-demand', methods=['GET'])
def skills_demand_page():
    return send_file(os.path.join(BASE_DIR, 'templates', 'skills_demand.html'))

@app.route('/api/top-skills', methods=['GET'])
def top_skills_data():
    try:
        skills = df['Skills Required'].dropna().tolist()
        all_skills = []
        for skill_list in skills:
            all_skills.extend([s.strip().lower() for s in skill_list.split(',') if s.strip()])

        top_counts = Counter(all_skills).most_common(10)
        labels = [item[0].capitalize() for item in top_counts]
        counts = [item[1] for item in top_counts]
        return jsonify({'labels': labels, 'counts': counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/skills-visualization', methods=['GET'])
def skills_visualization_page():
    return send_file(os.path.join(BASE_DIR, 'templates', 'skills_visualization.html'))

@app.route('/api/skill-cooccurrence', methods=['GET'])
def skill_cooccurrence_data():
    try:
        matrix_path = os.path.join(BASE_DIR, "static", "skill_cooccurrence_matrix.csv")
        network_path = os.path.join(BASE_DIR, "static", "skills_network.json")
        
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        skills = matrix_df.columns.tolist()
        matrix_values = matrix_df.values.tolist()
        
        with open(network_path, "r") as f:
            network_data = json.load(f)
        
        return jsonify({
            'skills': skills,
            'matrix': matrix_values,
            'network': network_data
        })
    except Exception as e:
        print("Error loading skill cooccurrence data:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/login')
def login():
    return send_file(os.path.join(BASE_DIR, 'templates', 'login.html'))

@app.route('/signup')
def signup():
    return send_file(os.path.join(BASE_DIR, 'templates', 'signup.html'))

@app.route('/salary-experience', methods=['GET'])
def salary_experience_page():
    return send_file(os.path.join(BASE_DIR, 'templates', 'salary_experience.html'))

@app.route('/api/salary-experience', methods=['GET'])
def salary_experience_data():
    try:
        data = df.copy()

        def extract_salary(s):
            nums = re.findall(r'\d+', str(s).replace(',', ''))
            nums = [int(n) for n in nums]
            if len(nums) == 0:
                return None
            return sum(nums) / len(nums)

        def extract_experience(s):
            nums = re.findall(r'\d+', str(s))
            nums = [int(n) for n in nums]
            if len(nums) == 0:
                return None
            return sum(nums) / len(nums)

        data['salary_num'] = data['Salary'].apply(extract_salary)
        data['experience_num'] = data['Experience'].apply(extract_experience)

        cleaned = data.dropna(subset=['salary_num', 'experience_num', 'Profile Name'])

        result = cleaned[[
            'Profile Name', 'Company Name', 'City',
            'salary_num', 'experience_num'
        ]].to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/salary-experience-box')
def salary_experience_box():
    try:
        data = df.copy()
        data = data.dropna(subset=['Experience', 'Salary', 'Profile Name'])

        def parse_exp(exp):
            parts = [int(s) for s in str(exp).split('-') if s.isdigit()]
            return sum(parts) / len(parts) if parts else None

        def parse_salary(sal):
            try:
                return int(str(sal).replace(',', '').split()[0])
            except:
                return None

        data['experience_num'] = data['Experience'].apply(parse_exp)
        data['salary_num'] = data['Salary'].apply(parse_salary)
        data.dropna(subset=['experience_num', 'salary_num'], inplace=True)

        data['experience_bin'] = pd.cut(data['experience_num'], 
                                     bins=[0,2,5,10,15,20,30], 
                                     labels=['0-2','2-5','5-10','10-15','15-20','20+'])

        out = data[['experience_bin', 'salary_num', 'Profile Name', 'Company Name', 'City']].to_dict(orient='records')
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/city-jobs-visualization')
def city_jobs_plot():
    try:
        with plot_lock:
            if df.empty:
                raise ValueError("No data available")
                
            df_clean = df.copy()
            df_clean['City'] = df_clean['City'].fillna('Unknown').str.strip()

            city_counts = df_clean['City'].value_counts().head(20)

            plt.figure(figsize=(12, 6))
            ax = city_counts.plot(kind='bar', color='#3f51b5')
            plt.title('Top 20 Cities by Job Openings', pad=20, fontsize=14)
            plt.xlabel('City', labelpad=10, fontsize=12)
            plt.ylabel('Number of Jobs', labelpad=10, fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.set_facecolor('#f5f7fa')
            plt.gcf().set_facecolor('#ffffff')
            
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=300, bbox_inches='tight', 
                       facecolor=plt.gcf().get_facecolor())
            img.seek(0)
            plt.close()

            return send_file(img, mimetype='image/png')
            
    except Exception as e:
        print(f"Error generating city jobs plot: {e}")
        blank_img = io.BytesIO()
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'Error loading data', ha='center', va='center')
        plt.savefig(blank_img, format='png')
        blank_img.seek(0)
        plt.close()
        return send_file(blank_img, mimetype='image/png')

@app.route('/city_jobs_page')
def city_jobs_page():
    try:
        return send_from_directory(os.path.join(BASE_DIR, 'templates'), 'city_jobs.html')
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True)