import requests
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk

# Initialize components
# nltk.download('punkt')
# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example job description
job_description = """Company Description

The Company: Prime Robotics is a leading global automation company serving the Logistics, Manufacturing and E-Commerce industries. We develop cutting-edge solutions that are innovative, productive, and scalable. Our mission is to accelerate supply chain logistics via best -in-class robotic solutions. We are in a startup mode, which means we are looking for someone with an entrepreneurial spirit who can tackle obstacles and not allow roadblocks, large or small to get in the way or slow down our processes. Is that you?

Prime Robotics is proud to be an EEOE, M/F/D/V, and we are committed to diversity both in practice and spirit at the corporate level.

Prime Robotics participates in E-Verify. E-Verify is an Internet-based system that compares information from an employee's Form I-9, Employment Eligibility Verification, to data from U.S. Department of Homeland Security and Social Security Administration records to confirm employment eligibility.

Job Description

You in The Role and on The Team: In this role you are SLAM Engineer with a background in Engineering and Robotics. Your duties may range from working on LiDAR-based SLAM, evaluating different sensors, visualizing data in order to make decisions on how to improve the system, and sensor fusion between LiDAR, and computer vision. Your experience includes having a deep understanding of the software and mechanics of Robotics including, Drivers, Camera’s, Lidars, sensors, autonomous vehicles, robots that move.

A Day in the Life
• Design, develop, implement, and optimize SLAM algorithms for computer vision and robotics systems
• Collaborate with cross-functional teams to integrate SLAM technology into our products.
• Work with cameras, IMUs, GPS, and other sensors to generate 3D maps and trajectories.
• Develop, test, and optimize SLAM backends using G2O or GTSAM
• Create and maintain code documentation, unit tests, and system test suites
• Explore new vision-based sensors and technologies
• Design custom architecture for vision-based autonomy
• Collaborate with cross-functional teams to integrate models and algorithms technology into our products
• Collaborate with cross-functional teams to integrate SLAM technology into our products.
• Work with cameras, IMUs, GPS, and other sensors to generate 3D maps and trajectories.
• Develop, test, and optimize SLAM backends using G2O or GTSAM
• Create and maintain code documentation, unit tests, and systems test suites

Qualifications

Key success factors
• 3+ years of experience, ideally in a robotics or autonomous systems field
• Hands on experience integrating sensors and algorithms on embedded systems
• Proficient in C++, Python, ROS2, and/or other robotics programming languages
• Strong background in SLAM, 3D reconstruction, Structure-from-Motion, Visual Inertial Odometry, and/or Bundle Adjustment
• Experience with SLAM backends such as G2O or GTSAM
• Strong foundations in multi-view geometry
• Experience in camera calibration
• Expert in complex rotations and frame transformations
• Strong problem-solving skills and ability to work in a fast-paced environment
• Strong verbal and written communication skills
• Experience with sensor fusion techniques to enhance positioning accuracy using data from various sensors like IMUs, LiDAR, and cameras.
• Experience developing scalable training pipelines in the cloud
• Experience with sensor fusion techniques to enhance positioning accuracy using data from various sensors like IMUs, LiDAR, and cameras
• Enthusiasm for the field of robotics and troubleshooting complex systems
• Bachelor’s Degree in Robotics, Mechanical or Electrical Engineering or relevant degree"""
# Tokenize the job description
tokens = word_tokenize(job_description)

# Part-of-speech tagging
tagged_tokens = pos_tag(tokens)

# Identify noun phrases
noun_phrases = []
current_phrase = []
for token, pos in tagged_tokens:
    if pos.startswith('NN'):  # NN for nouns
        current_phrase.append(token)
    else:
        if current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []

if current_phrase:  # In case the last word is a noun
    noun_phrases.append(' '.join(current_phrase))

print("Key topics in the job description:")
for phrase in noun_phrases:
    print(phrase)
# Function to fetch articles from Google search results
def fetch_google_articles(query, api_key, cx):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}&num=5"
    response = requests.get(url)
    data = response.json()
    articles = [(item['snippet'], item['link']) for item in data['items']]
    return articles
# Function to process articles
def process_articles(articles, job_description):
    relevant_sentences = []
    similarity_scores = []
    for snippet, link in articles:
        sentences = sent_tokenize(snippet)
        print("Number of articles:", len(articles))
        print("Articles:", articles)
        for sentence in sentences:
            sentiment_score = sia.polarity_scores(sentence)
            if sentiment_score['compound'] > 0.5:  # Filter positive sentiment sentences
                embedding = model.encode(sentence)
                job_embedding = model.encode(job_description)
                similarity_score = util.cos_sim(job_embedding, embedding).item()
                if similarity_score > 0.5:  # Filter relevant sentences
                    relevant_sentences.append(sentence)
                    similarity_scores.append(similarity_score)
    return relevant_sentences, similarity_scores

# Your Google Custom Search API key and cx
google_api_key = "AIzaSyBeKI6179qYKYQlKF0bZ97Lp4l2YfmdYnM"
google_cx = "f46a1f3417d09439b"


# Query for Google search
google_query = " ".join(noun_phrases) + " tutorial"

# Fetch Google articles
google_articles = fetch_google_articles(google_query, google_api_key, google_cx)
# print("Google Articles:", google_articles)

# Process Google articles
google_relevant_sentences, google_similarity_scores = process_articles(google_articles, job_description)

# Print comparison results
# Print comparison results
print("Number of Similarity Scores:", len(google_similarity_scores))
print("Google Articles Relevance:")
for i, (sentence, link) in enumerate(google_articles):
    print(f"URL: {link}")
    print(f"Sentence: {sentence}, Similarity: {google_similarity_scores[i]:.4f}")
    print()


