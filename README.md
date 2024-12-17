# Project Documentation: ChefMate Restaurant Clustering & Cooking Guide Application

## Project Overview
The **ChefMate** application aims to provide intelligent restaurant clustering, personalized recommendations based on user preferences, and a chatbot assistant specialized in cooking recipes. This project integrates machine learning for clustering cuisines and natural language processing for interactive guidance in cooking.

---

## Project Architecture
1. **Data Storage**:
   - pushed raw data to the s3 AWS service and retrived it for the next step
   - Cleaned and preprocessed data is stored as CSV files in the specified directory.
   - pushed the cleaned and structured data to the RDS AWS service and fetched it for the next step

2. **Clustering and Recommendation**:
   - Data is preprocessed and encoded.
   - PCA is applied for dimensionality reduction.
   - KMeans clustering groups restaurants into meaningful clusters.
   - Recommendations are generated based on user-input cuisines.

3. **Cooking Assistant**:
   - A chatbot is integrated using Google’s Generative AI.
   - The chatbot guides users with recipes and cooking tips based on natural language input.

4. **Streamlit Integration**:
   - Provides a user-friendly interface for clustering and chatbot functionalities.

5. **AWS service**:
   - used s3 for raw data storage
   - used RDS for the cleaned structured data storage
   - used EC2 for hosting the model on cloud

6. **Packages Used**:
   - pandas
   - boto3
   - sqlalchemy
   - psycopg2
   - os
   - matplotlib
   - seaborn
   - streamlit
   - scikit-learn
   - numpy
   - google-generativeai
   - python-dotenv
---

## Key Functionalities
### 1. **Restaurant Clustering and Recommendation**
#### Features:
- **Elbow Method**: Determines the optimal number of clusters.
- **Cluster Visualization**: Displays the distribution of cuisines in PCA-reduced space.
- **Personalized Recommendations**: Suggests restaurants based on user-preferred cuisines.

#### Implementation:
1. Data preprocessing and encoding.
2. PCA for dimensionality reduction.
3. KMeans clustering to identify cuisine groups.
4. Recommendation engine filters restaurants based on cluster labels.

### 2. **Cooking Assistant Chatbot**
#### Features:
- Responds to user queries about cooking, ingredients, and recipes.
- Provides step-by-step guidance for recipe preparation.
- Maintains context in the conversation using history.

#### Implementation:
1. Integrates Google Generative AI with API key authentication.
2. Defines a specialized system instruction to ensure responses are relevant to cooking.
3. Uses `start_chat` and `send_message` functions to manage interactions.

---

## Code Implementation
### 1. **Restaurant Clustering Module**
```python
import streamlit as st
from preprocess_model_training import CuisineClusteringModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Initialize model
model = CuisineClusteringModel('path_to_clean_data.csv')

def restaurant_clustering():
    st.title("Restaurant Clustering and Recommendations")
    
    # Preprocess data
    df_encoded = model.preprocess_data()

    # Perform PCA
    cuisines_pca = model.perform_pca()

    # Elbow Method
    st.subheader("Elbow Method for Optimal Clusters")
    inertia = []
    K = range(1, 15)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cuisines_pca)
        inertia.append(kmeans.inertia_)

    # Plot Elbow Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K, inertia, 'bx-')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal K')
    st.pyplot(fig)

    # Train Clustering Model
    optimal_k = st.slider("Select the optimal number of clusters", 2, 15, 5)
    df_encoded = model.train_kmeans(cuisines_pca, optimal_k)

    # Recommendations
    st.subheader("Restaurant Recommendations")
    user_input = st.text_input("Enter preferred cuisines (comma-separated):", "Indian, Pizza")
    preferred_cuisines = [c.strip() for c in user_input.split(',')]
    recommendations = model.recommend_restaurants(preferred_cuisines, top_n=10)
    st.write(f"### Top Restaurants for {', '.join(preferred_cuisines)}")
    st.write(recommendations)

    # Cluster Visualization
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x=cuisines_pca[:, 0],
        y=cuisines_pca[:, 1],
        hue=df_encoded['cluster'],
        palette='viridis',
        ax=ax
    )
    ax.set_title('Cuisine Clusters in PCA Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    st.pyplot(fig)
```

### 2. **Cooking Assistant Chatbot Module**
```python
import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant specialized in cooking recipes. You will only answer questions related to cooking, ingredients, and recipes. Do not provide any information outside the scope of cooking."
)

history = []  # Initialize chat history

st.write('Hello, how can I help you?')

user_input = st.text_input("You:")
if user_input:
    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    model_response = response.text
    st.write(f'Bot: {model_response}')
    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_response]})
```

---

## Setup and Configuration
1. **Environment Setup**:
    - Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
    - Create `.env` file for storing API keys:
      ```
      GOOGLE_API_KEY=your_google_api_key
      ```

2. **Folder Structure**:
    ```
    project_root/
    |-- converted_csv_data/
    |-- preprocess_model_training.py
    |-- app.py
    |-- requirements.txt
    |-- .env
    ```

3. **Run the Application**:
    ```bash
    streamlit run app.py
    ```

---

## Future Enhancements
1. Integrate additional clustering algorithms to improve recommendation accuracy.
2. Expand chatbot functionality to handle more complex recipe guidance.
3. Add dynamic map visualization for restaurant locations.
4. Optimize the backend by moving data to AWS RDS and S3 for scalability.

---
## Result
Created an app for the **Restaurant Recommendation System** using ML clustering with use of K-means clustering and had provided an optimal recommendation for the user based on the input the restaurant are clustered based on the cusinis and also integrated a **AI Chatbot** which acts as only a cooking guide assistant for the user where they can ask about all the cooking related stuff .Also the entire code run on the AWS service where i have used the **S3** for the raw file storage and retrived it from s3 cleraned the data and pushed the cleaned structured data to the **RDS**  ,retrived it for the futher ML training model and hosted it on the **EC2** 

---
