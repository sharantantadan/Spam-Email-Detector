import pandas as pd
import streamlit as st
import pickle
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Spam Email Detector", layout="wide", initial_sidebar_state="expanded")

# =================== TEXT PREPROCESSING ===================
def clean_text(text):
    """
    Preprocess email text:
    - Remove URLs
    - Convert to lowercase
    - Remove special characters
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# =================== DATA LOADING & MODEL TRAINING ===================
@st.cache_resource
def load_and_train_model():
    """
    Load data, train models, and return the best model with vectorizer and metrics.
    Cache this to avoid retraining on every Streamlit rerun.
    """
    
    # Try to load dataset with flexible path handling
    possible_paths = [
        "spam.csv",
        os.path.join(os.path.dirname(__file__), "spam.csv"),
        r"C:\Users\Likitha Shree\OneDrive\Desktop\spam_email\spam.csv",
        r"C:\Users\kband\OneDrive\Desktop\spam email\spam.csv"
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, encoding='latin-1')
                st.sidebar.success(f"✓ Dataset loaded from: {path}")
                break
            except Exception as e:
                st.sidebar.warning(f"Could not load from {path}: {e}")
    
    if data is None:
        st.error("❌ Could not find spam.csv. Please ensure the file exists in the project folder.")
        st.stop()
    
    # Keep only required columns
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data[['v1', 'v2']]
        data.columns = ['Category', 'Message']
    elif 'Category' in data.columns and 'Message' in data.columns:
        pass
    else:
        st.error("❌ Dataset must have columns: 'v1' and 'v2' (or 'Category' and 'Message')")
        st.stop()
    
    # Clean data
    data.drop_duplicates(inplace=True)
    data = data.dropna(subset=['Message', 'Category'])
    
    # Convert labels
    data['Category'] = data['Category'].str.lower()
    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
    
    # Apply text preprocessing
    data['Message_Cleaned'] = data['Message'].apply(clean_text)
    
    # Split data
    X = data['Message_Cleaned']
    y = data['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization (better than CountVectorizer)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train multiple models and select the best
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = 0
    model_scores = {}
    
    for model_name, model in models.items():
        model.fit(X_train_vec, y_train)
        score = model.score(X_test_vec, y_test)
        model_scores[model_name] = score
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Get predictions for metrics
    y_pred = best_model.predict(X_test_vec)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, labels=['Spam', 'Not Spam'], average='weighted'),
        'recall': recall_score(y_test, y_pred, labels=['Spam', 'Not Spam'], average='weighted'),
        'f1': f1_score(y_test, y_pred, labels=['Spam', 'Not Spam'], average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['Spam', 'Not Spam']),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return best_model, vectorizer, metrics, model_scores, X_test, y_test, X_test_vec, data

# =================== PREDICTION FUNCTION ===================
def predict_with_confidence(message, model, vectorizer):
    """
    Predict if a message is spam and return confidence score.
    """
    cleaned_message = clean_text(message)
    
    if not cleaned_message:
        return "Unable to process", 0
    
    message_vec = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vec)[0]
    
    # Get confidence (probability)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(message_vec)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = 0
    
    return prediction, confidence

# =================== DATASET ANALYSIS FUNCTION ===================
@st.cache_resource
def analyze_full_dataset(_model, _vectorizer):
    """
    Analyze entire dataset and categorize all messages.
    """
    # Load dataset
    possible_paths = [
        "spam.csv",
        os.path.join(os.path.dirname(__file__), "spam.csv"),
        r"C:\Users\Likitha Shree\OneDrive\Desktop\spam_email\spam.csv",
        r"C:\Users\kband\OneDrive\Desktop\spam email\spam.csv"
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, encoding='latin-1')
                break
            except:
                pass
    
    if data is None:
        return None
    
    # Handle column names
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data[['v1', 'v2']].copy()
        data.columns = ['Category', 'Message']
    elif 'Category' in data.columns and 'Message' in data.columns:
        data = data[['Category', 'Message']].copy()
    else:
        return None
    
    data = data.dropna(subset=['Message', 'Category'])
    
    # Predict on all messages
    data['Message_Cleaned'] = data['Message'].apply(clean_text)
    predictions = []
    confidences = []
    
    for msg in data['Message_Cleaned']:
        if msg.strip():
            msg_vec = _vectorizer.transform([msg])
            pred = _model.predict(msg_vec)[0]
            if hasattr(_model, 'predict_proba'):
                conf = max(_model.predict_proba(msg_vec)[0]) * 100
            else:
                conf = 0
            predictions.append(pred)
            confidences.append(conf)
        else:
            predictions.append('Unable to process')
            confidences.append(0)
    
    data['Predicted_Category'] = predictions
    data['Confidence'] = confidences
    
    # Analyze composition
    data['Message_Length'] = data['Message'].apply(lambda x: len(str(x)))
    data['Word_Count'] = data['Message_Cleaned'].apply(lambda x: len(str(x).split()))
    data['Original_Category'] = data['Category'].str.lower()
    
    return data

def get_top_words(messages, top_n=15):
    """
    Extract top N most common words from messages.
    """
    from collections import Counter
    words = []
    for msg in messages:
        words.extend(str(msg).split())
    word_counts = Counter(words)
    return dict(word_counts.most_common(top_n))

def get_spam_composition_stats(data):
    """
    Get detailed statistics about spam composition.
    """
    stats = {}
    
    # Overall stats
    total_msgs = len(data)
    spam_msgs = len(data[data['Predicted_Category'] == 'Spam'])
    ham_msgs = len(data[data['Predicted_Category'] == 'Not Spam'])
    
    stats['total_messages'] = total_msgs
    stats['spam_count'] = spam_msgs
    stats['ham_count'] = ham_msgs
    stats['spam_percentage'] = (spam_msgs / total_msgs * 100) if total_msgs > 0 else 0
    
    # Length statistics
    spam_data = data[data['Predicted_Category'] == 'Spam']
    ham_data = data[data['Predicted_Category'] == 'Not Spam']
    
    stats['avg_spam_length'] = spam_data['Message_Length'].mean() if len(spam_data) > 0 else 0
    stats['avg_ham_length'] = ham_data['Message_Length'].mean() if len(ham_data) > 0 else 0
    stats['avg_spam_words'] = spam_data['Word_Count'].mean() if len(spam_data) > 0 else 0
    stats['avg_ham_words'] = ham_data['Word_Count'].mean() if len(ham_data) > 0 else 0
    
    return stats

# =================== STREAMLIT UI ===================
st.title("🚀 Advanced Spam Email Detector")
st.markdown("---")

# Load model and data
model, vectorizer, metrics, model_scores, X_test, y_test, X_test_vec, full_data = load_and_train_model()

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col2:
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1-Score", f"{metrics['f1']:.2%}")
    
    st.divider()
    
    st.subheader("All Models Tested")
    for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
        st.write(f"**{model_name}**: {score:.2%}")
    
    st.divider()
    
    st.subheader("Dataset Info")
    st.write(f"**Total Messages**: {len(full_data)}")
    spam_count = (full_data['Category'] == 'Spam').sum()
    st.write(f"**Spam Messages**: {spam_count}")
    st.write(f"**Legitimate Messages**: {len(full_data) - spam_count}")

# Main Content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detector", "📊 Spam Composition", "📈 Analytics", "ℹ️ About"])

with tab1:
    st.header("Test Your Message")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        input_message = st.text_area(
            "Enter email message or text:",
            height=120,
            placeholder="Paste your email content here..."
        )
    
    with col2:
        st.write("")
        st.write("")
        detect_button = st.button("🔍 Check Message", use_container_width=True, type="primary")
    
    if detect_button:
        if input_message.strip() == "":
            st.warning("⚠️ Please enter a message to analyze")
        else:
            prediction, confidence = predict_with_confidence(input_message, model, vectorizer)
            
            # Display result with color coding
            if prediction == "Spam":
                st.error(f"🚨 **SPAM DETECTED**")
                st.metric("Spam Confidence", f"{confidence:.1f}%")
            else:
                st.success(f"✅ **LEGITIMATE EMAIL**")
                st.metric("Legitimacy Confidence", f"{confidence:.1f}%")
            
            # Show cleaned version of text
            with st.expander("🔍 See preprocessed text"):
                cleaned = clean_text(input_message)
                st.write(cleaned if cleaned else "No content after preprocessing")

with tab2:
    st.header("📊 Spam Composition Analysis")
    st.markdown("Analyzing entire dataset and categorizing all spam messages...")
    
    # Use the already loaded full dataset and analyze it
    if full_data is not None:
        # Predict on all messages
        predictions = []
        confidences = []
        
        for msg in full_data['Message_Cleaned']:
            if msg.strip():
                msg_vec = vectorizer.transform([msg])
                pred = model.predict(msg_vec)[0]
                if hasattr(model, 'predict_proba'):
                    conf = max(model.predict_proba(msg_vec)[0]) * 100
                else:
                    conf = 0
                predictions.append(pred)
                confidences.append(conf)
            else:
                predictions.append('Unable to process')
                confidences.append(0)
        
        dataset_analysis = full_data.copy()
        dataset_analysis['Predicted_Category'] = predictions
        dataset_analysis['Confidence'] = confidences
        
        # Analyze composition
        dataset_analysis['Message_Length'] = dataset_analysis['Message'].apply(lambda x: len(str(x)))
        dataset_analysis['Word_Count'] = dataset_analysis['Message_Cleaned'].apply(lambda x: len(str(x).split()))
        dataset_analysis['Original_Category'] = dataset_analysis['Category'].str.lower()
        # Get statistics
        stats = get_spam_composition_stats(dataset_analysis)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📧 Total Messages", f"{stats['total_messages']}")
        with col2:
            st.metric("🚨 Spam Messages", f"{stats['spam_count']} ({stats['spam_percentage']:.1f}%)")
        with col3:
            st.metric("✅ Legitimate", f"{stats['ham_count']}")
        with col4:
            st.metric("🎯 Spam Ratio", f"1 in {stats['total_messages']//stats['spam_count'] if stats['spam_count'] > 0 else 0}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution: Spam vs Legitimate")
            category_counts = pd.DataFrame({
                'Category': ['Spam', 'Legitimate'],
                'Count': [stats['spam_count'], stats['ham_count']]
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#FF6B6B', '#4CAF50']
            wedges, texts, autotexts = ax.pie(
                category_counts['Count'],
                labels=category_counts['Category'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            ax.set_title('Message Distribution', fontsize=14, weight='bold', pad=20)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Average Message Length")
            length_data = pd.DataFrame({
                'Type': ['Spam', 'Legitimate'],
                'Avg Length': [stats['avg_spam_length'], stats['avg_ham_length']]
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(length_data['Type'], length_data['Avg Length'], color=['#FF6B6B', '#4CAF50'], width=0.6)
            ax.set_ylabel('Characters', fontsize=11, weight='bold')
            ax.set_title('Average Message Length Comparison', fontsize=14, weight='bold', pad=20)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Word Count")
            word_data = pd.DataFrame({
                'Type': ['Spam', 'Legitimate'],
                'Avg Words': [stats['avg_spam_words'], stats['avg_ham_words']]
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(word_data['Type'], word_data['Avg Words'], color=['#FF6B6B', '#4CAF50'], width=0.6)
            ax.set_ylabel('Words', fontsize=11, weight='bold')
            ax.set_title('Average Words per Message', fontsize=14, weight='bold', pad=20)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Message Length Distribution")
            spam_data = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Spam']['Message_Length']
            ham_data = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Not Spam']['Message_Length']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist([spam_data, ham_data], bins=30, label=['Spam', 'Legitimate'], color=['#FF6B6B', '#4CAF50'], alpha=0.7)
            ax.set_xlabel('Message Length (characters)', fontsize=11, weight='bold')
            ax.set_ylabel('Frequency', fontsize=11, weight='bold')
            ax.set_title('Message Length Distribution', fontsize=14, weight='bold', pad=20)
            ax.legend()
            st.pyplot(fig)
        
        st.divider()
        
        # Top words analysis
        st.subheader("📝 Most Common Words")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 15 Words in Spam Messages**")
            spam_messages = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Spam']['Message_Cleaned']
            spam_words = get_top_words(spam_messages, top_n=15)
            
            if spam_words:
                spam_df = pd.DataFrame(list(spam_words.items()), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(spam_df['Word'], spam_df['Frequency'], color='#FF6B6B')
                ax.set_xlabel('Frequency', fontweight='bold')
                ax.set_title('Top Words in Spam', fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.warning("No spam messages to analyze")
        
        with col2:
            st.write("**Top 15 Words in Legitimate Messages**")
            ham_messages = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Not Spam']['Message_Cleaned']
            ham_words = get_top_words(ham_messages, top_n=15)
            
            if ham_words:
                ham_df = pd.DataFrame(list(ham_words.items()), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(ham_df['Word'], ham_df['Frequency'], color='#4CAF50')
                ax.set_xlabel('Frequency', fontweight='bold')
                ax.set_title('Top Words in Legitimate', fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.warning("No legitimate messages to analyze")
        
        st.divider()
        
        # Model confidence analysis
        st.subheader("🎯 Prediction Confidence Analysis")
        
        spam_confidence = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Spam']['Confidence']
        ham_confidence = dataset_analysis[dataset_analysis['Predicted_Category'] == 'Not Spam']['Confidence']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist([spam_confidence, ham_confidence], bins=20, label=['Spam', 'Legitimate'], color=['#FF6B6B', '#4CAF50'], alpha=0.7)
        ax.set_xlabel('Confidence Score (%)', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Model Prediction Confidence Distribution', fontsize=14, weight='bold', pad=20)
        ax.legend()
        ax.axvline(80, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
        st.pyplot(fig)
        
        st.divider()
        
        # Detailed statistics table
        st.subheader("📋 Detailed Statistics")
        
        detailed_stats = {
            'Metric': [
                'Total Messages',
                'Spam Count',
                'Legitimate Count',
                'Spam Percentage',
                'Avg Spam Length (chars)',
                'Avg Legitimate Length (chars)',
                'Avg Spam Words',
                'Avg Legitimate Words',
                'Min Spam Length',
                'Max Spam Length'
            ],
            'Value': [
                f"{stats['total_messages']}",
                f"{stats['spam_count']}",
                f"{stats['ham_count']}",
                f"{stats['spam_percentage']:.2f}%",
                f"{stats['avg_spam_length']:.0f}",
                f"{stats['avg_ham_length']:.0f}",
                f"{stats['avg_spam_words']:.2f}",
                f"{stats['avg_ham_words']:.2f}",
                f"{dataset_analysis[dataset_analysis['Predicted_Category'] == 'Spam']['Message_Length'].min() if stats['spam_count'] > 0 else 0}",
                f"{dataset_analysis[dataset_analysis['Predicted_Category'] == 'Spam']['Message_Length'].max() if stats['spam_count'] > 0 else 0}"
            ]
        }
        
        stats_df = pd.DataFrame(detailed_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # View categorized dataset
        st.subheader("📊 View Categorized Messages")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            category_filter = st.selectbox(
                "Filter by category:",
                ["All", "Spam", "Legitimate"]
            )
        
        with filter_col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Confidence (High to Low)", "Confidence (Low to High)", "Message Length (Long to Short)"]
            )
        
        # Apply filters
        filtered_data = dataset_analysis.copy()
        
        if category_filter == "Spam":
            filtered_data = filtered_data[filtered_data['Predicted_Category'] == 'Spam']
        elif category_filter == "Legitimate":
            filtered_data = filtered_data[filtered_data['Predicted_Category'] == 'Not Spam']
        
        # Apply sorting
        if sort_by == "Confidence (High to Low)":
            filtered_data = filtered_data.sort_values('Confidence', ascending=False)
        elif sort_by == "Confidence (Low to High)":
            filtered_data = filtered_data.sort_values('Confidence', ascending=True)
        elif sort_by == "Message Length (Long to Short)":
            filtered_data = filtered_data.sort_values('Message_Length', ascending=False)
        
        # Display table
        display_cols = ['Predicted_Category', 'Message', 'Confidence', 'Message_Length', 'Word_Count']
        display_df = filtered_data[display_cols].head(100).reset_index(drop=True)
        display_df.columns = ['Category', 'Message', 'Confidence (%)', 'Length', 'Words']
        display_df['Confidence (%)'] = display_df['Confidence (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        csv = dataset_analysis.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Analysis as CSV",
            data=csv,
            file_name="spam_analysis.csv",
            mime="text/csv"
        )
    else:
        st.error("Could not load dataset for analysis")

with tab3:
    st.header("Model Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Spam', 'Not Spam'],
            yticklabels=['Spam', 'Not Spam'],
            ax=ax
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Model Comparison")
        model_data = pd.DataFrame(list(model_scores.items()), columns=['Model', 'Accuracy'])
        model_data = model_data.sort_values('Accuracy', ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(model_data['Model'], model_data['Accuracy'], color='steelblue')
        ax.set_xlabel('Accuracy Score')
        ax.set_xlim(0.9, 1.0)
        for i, v in enumerate(model_data['Accuracy']):
            ax.text(v - 0.01, i, f'{v:.2%}', va='center', ha='right', fontweight='bold')
        st.pyplot(fig)

with tab4:
    st.subheader("About This Detector")
    st.markdown("""
    ### Features
    - **Advanced Text Preprocessing**: Removes URLs, emails, special characters
    - **TF-IDF Vectorization**: Better feature extraction than CountVectorizer
    - **Multiple Models**: Trains Logistic Regression, Naive Bayes, and Random Forest
    - **Confidence Scores**: Shows probability of prediction
    - **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
    
    ### How It Works
    1. **Preprocessing**: Cleans and normalizes incoming messages
    2. **Vectorization**: Converts text to numerical features using TF-IDF
    3. **Classification**: Uses the best trained model to predict
    4. **Confidence**: Returns probability score for the prediction
    
    ### Best Practices
    - Longer messages provide more reliable predictions
    - The model learns from the spam.csv dataset provided
    - Confidence scores > 80% are highly reliable
    
    **Built with**: Streamlit, scikit-learn, Pandas
    """)