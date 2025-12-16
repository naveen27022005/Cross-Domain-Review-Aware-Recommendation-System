

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .rating-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
    }
    .movie-description {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        line-height: 1.4;
    }
    .movie-meta {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    .movie-id-badge {
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.75rem;
        font-family: monospace;
        display: inline-block;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# MODEL ARCHITECTURE


class BranchBModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, shared_dim=128, dropout=0.4):
        super().__init__()
        
        self.user_shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.user_specific_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.user_decoder = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_vector, item_vector):
        user_shared = self.user_shared_encoder(user_vector)
        user_specific = self.user_specific_encoder(user_vector)
        
        user_combined = torch.cat([user_shared, user_specific], dim=1)
        user_reconstructed = self.user_decoder(user_combined)
        
        item_features = self.item_encoder(item_vector)
        
        combined_features = torch.cat([user_shared, item_features], dim=1)
        rating_raw = self.predictor(combined_features).squeeze(-1)
        
        rating_pred = rating_raw * 4.0 + 1.0
        
        return rating_pred, user_shared, user_specific, user_reconstructed


# LOAD MODELS AND DATA


@st.cache_resource
def load_bert_model():
    """Load DistilBERT for encoding reviews"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    bert_model.eval()
    return tokenizer, bert_model

@st.cache_resource
def load_recommendation_model():
    """Load trained Branch B model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BranchBModel(
        input_dim=768,
        hidden_dim=256,
        shared_dim=128,
        dropout=0.4
    ).to(device)
    
    try:
        checkpoint = torch.load('best_model_fixed.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device, checkpoint
    except FileNotFoundError:
        st.error(" Model file 'best_model_fixed.pt' not found!")
        return None, device, None

@st.cache_resource
def load_movie_embeddings():
    """Load pre-computed movie embeddings"""
    try:
        with open('data/embeddings/test_item_embeddings.pkl', 'rb') as f:
            movie_embeddings = pickle.load(f)
        return movie_embeddings
    except FileNotFoundError:
        st.error("⚠️ Movie embeddings not found at 'data/embeddings/test_item_embeddings.pkl'")
        return None

@st.cache_data
def load_movie_metadata(metadata_path='meta_Movies_and_TV.jsonl'):
   
    movie_metadata = {}
    missing_titles_count = 0
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    movie = json.loads(line.strip())
                    movie_id = movie.get('parent_asin')
                    
                    if movie_id:
                        # Get title with fallback
                        title = movie.get('title')
                        if not title or str(title).strip() in ['', 'None', 'null']:
                            title = None
                            missing_titles_count += 1
                        
                        # Get description with proper handling
                        description = 'No description available'
                        if movie.get('description'):
                            if isinstance(movie['description'], list) and len(movie['description']) > 0:
                                desc_text = movie['description'][0]
                                # Clean HTML from description
                                if '<div' in str(desc_text) or '<p>' in str(desc_text):
                                    # Skip HTML-heavy descriptions
                                    description = 'No description available'
                                else:
                                    description = desc_text
                            elif isinstance(movie['description'], str):
                                description = movie['description']
                        
                        # Extract key information with safe defaults
                        movie_metadata[movie_id] = {
                            'title': title,
                            'description': description,
                            'categories': movie.get('categories', []),
                            'average_rating': movie.get('average_rating'),
                            'rating_number': movie.get('rating_number'),
                            'features': movie.get('features', []),
                            'price': movie.get('price'),
                            'main_category': movie.get('main_category', 'Movies & TV')
                        }
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        st.success(f" Loaded metadata for {len(movie_metadata):,} movies.))")
        return movie_metadata
        
    except FileNotFoundError:
        st.warning(f"⚠️ Metadata file '{metadata_path}' not found. Using movie IDs only.")
        return {}


# ENCODING FUNCTIONS


def encode_reviews(reviews: List[str], tokenizer, bert_model) -> torch.Tensor:
   
    embeddings = []
    
    with torch.no_grad():
        for review in reviews:
            # Tokenize
            inputs = tokenizer(
                review,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get BERT embeddings
            outputs = bert_model(**inputs)
            
            # Mean pooling over tokens
            embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
            embeddings.append(embedding)
    
    # Aggregate all reviews (mean pooling)
    if embeddings:
        aggregated = torch.cat(embeddings, dim=0).mean(dim=0)  # [768]
        return aggregated
    else:
        return torch.zeros(768)

def get_recommendations(
    user_reviews: List[str],
    movie_embeddings: Dict,
    movie_metadata: Dict,
    model,
    tokenizer,
    bert_model,
    device,
    top_k: int = 10,
    min_rating: float = 3.5
) -> List[Dict]:
   
    
    # 1. Encode user reviews
    user_embedding = encode_reviews(user_reviews, tokenizer, bert_model)
    user_vector = user_embedding.unsqueeze(0).to(device)  # [1, 768]
    
    # 2. Predict ratings for all movies
    recommendations = []
    
    with torch.no_grad():
        for movie_id, movie_emb in movie_embeddings.items():
            # Convert movie embedding to tensor
            if isinstance(movie_emb, np.ndarray):
                movie_emb = torch.from_numpy(movie_emb).float()
            
            if movie_emb.dim() == 1:
                movie_emb = movie_emb.unsqueeze(0)
            
            movie_vector = movie_emb.mean(dim=0).unsqueeze(0).to(device)  # [1, 768]
            
            # Predict rating
            rating_pred, _, _, _ = model(user_vector, movie_vector)
            predicted_rating = rating_pred.item()
            
            # Filter by minimum rating
            if predicted_rating >= min_rating:
                # Get movie metadata
                metadata = movie_metadata.get(movie_id, {})
                
                # Handle title - use movie_id if title is None or missing
                title = metadata.get('title')
                if not title or title == 'None' or str(title).strip() == '' or title == 'null':
                    # Try to find in original data or use ID
                    title = f"[Movie ID: {movie_id}]"
                
                # Handle description
                description = metadata.get('description', 'No description available')
                if not description or description == 'None':
                    description = 'No description available'
                
                # Handle categories
                categories = metadata.get('categories', [])
                if not categories or categories == ['None']:
                    categories = []
                
                recommendations.append({
                    'movie_id': movie_id,
                    'title': title,
                    'description': description,
                    'categories': categories,
                    'average_rating': metadata.get('average_rating'),
                    'rating_number': metadata.get('rating_number'),
                    'features': metadata.get('features', []),
                    'predicted_rating': predicted_rating,
                    'main_category': metadata.get('main_category', 'Movies & TV')
                })
    
    # 3. Sort by rating (descending)
    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # 4. Return top K
    return recommendations[:top_k]


# STREAMLIT APP


def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get personalized movie recommendations based on your book preferences!</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('Loading models... This may take a moment on first run.'):
        tokenizer, bert_model = load_bert_model()
        model, device, checkpoint = load_recommendation_model()
        movie_embeddings = load_movie_embeddings()
        movie_metadata = load_movie_metadata()  
    
    if model is None or movie_embeddings is None:
        st.error(" Failed to load required models or data. Please check file paths.")
        return
    
    # Show model info
    with st.sidebar:
        st.markdown("---")
        st.header(" Settings")
        top_k = st.slider("Number of recommendations", 5, 20, 10)
        min_rating = st.slider("Minimum rating threshold", 1.0, 5.0, 3.5, 0.1)
        show_only_with_titles = st.checkbox("Only show movies with titles", value=True)
    
    # Main content
    st.markdown("##  Tell us about books you've enjoyed")
    
    st.markdown('<div class="info-box">💡 <b>Tip:</b> Add reviews of books you loved or hated. The more detail, the better the recommendations!</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'book_reviews' not in st.session_state:
        st.session_state.book_reviews = [
            {'book': '', 'review': '', 'rating': 5}
        ]
    
    # Book review inputs
    for idx, review_data in enumerate(st.session_state.book_reviews):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 6, 2, 1])
            
            with col1:
                book = st.text_input(
                    "Book Title",
                    value=review_data['book'],
                    key=f"book_{idx}",
                    placeholder="e.g., Harry Potter"
                )
            
            with col2:
                review = st.text_area(
                    "Your Review",
                    value=review_data['review'],
                    key=f"review_{idx}",
                    placeholder="What did you think? Be descriptive!",
                    height=100
                )
            
            with col3:
                rating = st.select_slider(
                    "Your Rating",
                    options=[1, 2, 3, 4, 5],
                    value=review_data['rating'],
                    key=f"rating_{idx}"
                )
            
            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"delete_{idx}"):
                    if len(st.session_state.book_reviews) > 1:
                        st.session_state.book_reviews.pop(idx)
                        st.rerun()
            
            # Update session state
            st.session_state.book_reviews[idx] = {
                'book': book,
                'review': review,
                'rating': rating
            }
        
        st.markdown("---")
    
    # Add review button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➕ Add Another Book", use_container_width=True):
            st.session_state.book_reviews.append({
                'book': '',
                'review': '',
                'rating': 5
            })
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Get recommendations button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        recommend_button = st.button("🎬 Get Movie Recommendations", use_container_width=True, type="primary")
    
    # Generate recommendations
    if recommend_button:
        # Validate input
        valid_reviews = [
            r['review'] for r in st.session_state.book_reviews 
            if r['book'].strip() and r['review'].strip()
        ]
        
        if len(valid_reviews) == 0:
            st.error(" Please add at least one book review before getting recommendations!")
        else:
            with st.spinner('🔮 Analyzing your preferences and generating recommendations...'):
                # Get recommendations
                recommendations = get_recommendations(
                    user_reviews=valid_reviews,
                    movie_embeddings=movie_embeddings,
                    movie_metadata=movie_metadata,
                    model=model,
                    tokenizer=tokenizer,
                    bert_model=bert_model,
                    device=device,
                    top_k=top_k * 2, 
                    min_rating=min_rating
                )
                
                # Filter out movies without proper titles if checkbox is enabled
                if show_only_with_titles:
                    recommendations = [
                        r for r in recommendations 
                        if r['title'] and not r['title'].startswith('[Movie ID:')
                    ][:top_k]
                else:
                    recommendations = recommendations[:top_k]
                
                if len(recommendations) == 0:
                    st.warning(f"No movies found with rating ≥ {min_rating:.1f}. Try lowering the threshold.")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##  Your Personalized Movie Recommendations")
                    st.markdown(f"*Based on {len(valid_reviews)} book review(s)*")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display recommendations
                    for idx, rec in enumerate(recommendations, 1):
                        rating = rec['predicted_rating']
                        stars = "⭐" * int(round(rating))
                        
                        # Safe handling of categories
                        categories = rec.get('categories', [])
                        if categories and len(categories) > 0:
                            categories_str = ", ".join(str(c) for c in categories[:3] if c)
                        else:
                            categories_str = rec.get('main_category', 'N/A')
                        
                        # Safe handling of features
                        features = rec.get('features', [])
                        if features and len(features) > 0:
                            features_str = " | ".join(str(f) for f in features[:3] if f)
                        else:
                            features_str = ""
                        
                        # Build metadata line
                        meta_parts = []
                        if rec.get('average_rating'):
                            try:
                                avg_rating = float(rec['average_rating'])
                                meta_parts.append(f"⭐ {avg_rating:.1f}/5")
                            except (ValueError, TypeError):
                                pass
                        
                        if rec.get('rating_number'):
                            try:
                                rating_num = int(rec['rating_number'])
                                meta_parts.append(f"({rating_num:,} ratings)")
                            except (ValueError, TypeError):
                                pass
                        
                        if features_str:
                            meta_parts.append(features_str)
                        
                        meta_line = " • ".join(meta_parts) if meta_parts else ""
                        
                        # Safe description handling
                        description = rec.get('description', 'No description available')
                        if not description or description == 'None':
                            description = 'No description available'
                        
                        # Escape HTML characters in description to prevent rendering issues
                        import html
                        description_escaped = html.escape(str(description))
                        
                        # Truncate description
                        description_display = description_escaped[:300] + '...' if len(description_escaped) > 300 else description_escaped
                        
                        # Escape title and categories to prevent HTML rendering
                        title_escaped = html.escape(str(rec['title']))
                        categories_escaped = html.escape(str(categories_str))
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3>#{idx} {title_escaped} <span class="movie-id-badge">ID: {rec['movie_id']}</span></h3>
                            <div class="rating-badge">Predicted: {rating:.2f} {stars}</div>
                            <div class="movie-meta">
                                <b>Category:</b> {categories_escaped}
                            </div>
                            {f'<div class="movie-meta">{html.escape(meta_line)}</div>' if meta_line else ''}
                            <div class="movie-description">
                                {description_display}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Statistics
                    st.markdown("<br>", unsafe_allow_html=True)
                    avg_rating = np.mean([r['predicted_rating'] for r in recommendations])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(" Recommendations", len(recommendations))
                    with col2:
                        st.metric(" Average Rating", f"{avg_rating:.2f}")
                    with col3:
                        st.metric(" Threshold", f"{min_rating:.1f}+")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><b>Cross-Domain Recommendation System</b></p>
        <p>Transferring preferences from Books → Movies using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()