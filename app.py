import streamlit as st
from preprocess_model_training import CuisineClusteringModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# Initialize model
model = CuisineClusteringModel('clean_data.csv')

# Streamlit Pages
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
    recommendations = model.recommend_restaurants(preferred_cuisines, top_n=100000)
    st.write(f"### Top Restaurants for {', '.join(preferred_cuisines)}")
     # Filters
    city = st.selectbox("Select City", sorted(recommendations['city'].unique()), index=0)
    min_cost, max_cost = st.slider("Filter by Average Cost for Two", 0, 5000, (100, 1500), step=100)
    min_rating, max_rating = st.slider("Filter by Rating", 0.0, 5.0, (3.5, 5.0), step=0.1)
    has_online_delivery = st.checkbox("Has Online Delivery")

    # Apply filters
    filtered_df = recommendations[
        (recommendations['city'] == city) &
        (recommendations['average_cost_for_two'] >= min_cost) &
        (recommendations['average_cost_for_two'] <= max_cost) &
        (recommendations['rating'] >= min_rating) &
        (recommendations['rating'] <= max_rating)
    ]

    if has_online_delivery:
        filtered_df = filtered_df[filtered_df['has_online_delivery'] == 1]
        
    st.write(filtered_df)

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

# Main App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Restaurant Clustering","Chat Bot(Cooking Assistant)"])


if page == "Restaurant Clustering":
    restaurant_clustering()
elif page == "Chat Bot(Cooking Assistant)":

    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    st.title('🤖 Chatbot')
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    # User input
    if prompt := st.chat_input():
        try:
            generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",}
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config=generation_config,
                system_instruction=(
                    "You are a helpful assistant specialized in cooking recipes. "
                    "You will only answer questions related to cooking, ingredients, and recipes. "
                    "Do not provide any information outside the scope of cooking."
                ),
            )
            chat = model.start_chat(history=[])

            response = chat.send_message(prompt, stream=True)
            assistant_response = ''
            for chunk in response:
                assistant_response += chunk.text

            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        st.rerun()
        

    
