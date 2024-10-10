import os
import time
import logging
import requests
from flask import Flask, request, jsonify
from flask_caching import Cache
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY is not set in the environment variables.")

# Initialize Flask app
app = Flask(__name__)

# Configure Flask-Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def construct_url(query, page=1, page_size=20):
    """
    Constructs the NewsAPI URL with proper URL encoding.
    
    Args:
        query (str): The search query.
        page (int): The page number for pagination.
        page_size (int): Number of articles per page.
        
    Returns:
        str: The fully constructed and encoded URL.
    """
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "page": page,
        "pageSize": page_size,
        "language": "en",  # Optional: Specify language
        "sortBy": "publishedAt"  # Optional: Sort by publication date
    }
    url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
    logger.debug(f"Constructed URL: {url}")
    return url

def fetch_news(query="latest", page=1, page_size=20, max_retries=3):
    """
    Fetches news articles from NewsAPI based on the provided query.
    Implements pagination, rate limiting, and input validation.
    
    Args:
        query (str): The search query.
        page (int): The page number for pagination.
        page_size (int): Number of articles per page.
        max_retries (int): Maximum number of retries for rate limiting.
        
    Returns:
        list: A list of dictionaries containing news articles.
    """
    # Input Validation and Sanitization
    if not isinstance(query, str) or not query.strip():
        logger.warning("Invalid query parameter provided. Defaulting to 'latest'.")
        query = "latest"
    
    url = construct_url(query, page, page_size)
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }
    
    retries = 0
    backoff_time = 1  # Initial backoff time in seconds
    
    while retries < max_retries:
        try:
            logger.info(f"Fetching news for query: '{query}', Page: {page}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            
            if news_data.get("status") != "ok":
                logger.error(f"NewsAPI returned non-ok status: {news_data.get('status')}")
                return []
            
            articles = news_data.get("articles", [])
            logger.info(f"Fetched {len(articles)} articles from page {page}.")
            return parse_articles(articles)
        
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                # Too Many Requests - Rate Limiting
                retry_after = int(response.headers.get("Retry-After", backoff_time))
                logger.warning(f"Rate limited by NewsAPI. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                retries += 1
                backoff_time *= 2  # Exponential backoff
            else:
                logger.error(f"HTTP error occurred: {http_err}")
                return []
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request exception occurred: {req_err}")
            return []
        except ValueError as val_err:
            logger.error(f"JSON decoding failed: {val_err}")
            return []
    
    logger.error("Max retries exceeded. Failed to fetch news.")
    return []

def parse_articles(articles):
    """
    Transforms raw articles into a consistent structured format.
    
    Args:
        articles (list): List of articles from NewsAPI.
        
    Returns:
        list: List of dictionaries with selected article fields.
    """
    structured_articles = []
    for article in articles:
        structured_article = {
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
            "publishedAt": article.get("publishedAt"),
            "source": article.get("source", {}).get("name")
        }
        structured_articles.append(structured_article)
    logger.debug(f"Parsed {len(structured_articles)} articles.")
    return structured_articles

@app.route('/news', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes
def index():
    """
    Flask route to fetch and return news articles based on query parameters.
    
    Query Parameters:
        query (str): The search query (default: 'latest').
        page (int): The page number for pagination (default: 1).
        page_size (int): Number of articles per page (default: 20).
        
    Returns:
        JSON: A JSON response containing a list of news articles.
    """
    query = request.args.get("query", "latest")
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 20, type=int)
    
    if page_size > 100:
        logger.warning("Page size exceeds maximum limit. Setting to 100.")
        page_size = 100  # NewsAPI max page size is 100
    
    articles = fetch_news(query=query, page=page, page_size=page_size)
    return jsonify(articles)

if __name__ == "__main__":
    # Example usage for standalone testing
    test_query = "COVID-19"
    test_articles = fetch_news(test_query, page=1, page_size=5)
    for article in test_articles:
        print(f"Title: {article['title']}\nURL: {article['url']}\n")
