import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import logging
import time

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def construct_url(location, specialization, page=1):
    """
    Constructs the Practo search URL using proper URL encoding.
    
    Args:
        location (str): The city/location to search in.
        specialization (str): The doctor's specialization.
        page (int): The page number for pagination.
    
    Returns:
        str: The fully constructed and encoded URL.
    """
    query = {
        "results_type": "doctor",
        "q": f'[{{"word":"{specialization}","autocompleted":true,"category":"subspeciality"}}]',
        "city": location,
        "page": page  # Assuming Practo uses 'page' for pagination
    }
    encoded_query = urlencode(query, safe='[]{}":,')
    url = f'https://www.practo.com/search/doctors?{encoded_query}'
    logger.debug(f'Constructed URL: {url}')
    return url

def fetch_page(url, headers):
    """
    Fetches the content of a given URL with specified headers.
    
    Args:
        url (str): The URL to fetch.
        headers (dict): The HTTP headers to include in the request.
    
    Returns:
        BeautifulSoup object or None: Parsed HTML content or None if an error occurs.
    """
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f'Successfully fetched data from: {url}')
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        logger.error(f'Error fetching data from Practo: {e}')
        return None

def parse_doctor_info(doctor_card):
    """
    Parses the HTML content to extract doctor's information.
    
    Args:
        doctor_card (bs4.element.Tag): The HTML element containing doctor's data.
    
    Returns:
        dict: A dictionary containing doctor's details.
    """
    doctor_info = {}

    # Extracting doctor's name
    name = doctor_card.find('h2', class_="doctor-name")
    doctor_info['Name'] = name.text.strip() if name else ""

    # Extracting link to doctor's profile
    link = doctor_card.find('a', href=True)
    doctor_info['Profile Link'] = f"https://www.practo.com{link['href']}" if link and link['href'] else ""

    # Extracting specialization
    specialization_elem = doctor_card.find('span')
    doctor_info['Specialization'] = specialization_elem.text.strip() if specialization_elem else ""

    # Extracting experience
    experience_elem = doctor_card.find('div', class_="uv2-spacer--xs-top")
    experience_text = experience_elem.text.strip().replace('\xa0', ' ') if experience_elem else ""
    doctor_info['Experience'] = experience_text

    # Extracting location
    loc_elem = doctor_card.find('div', class_="u-bold u-d-inlineblock u-valign--middle")
    location_parts = loc_elem.find_all('span') if loc_elem else []
    doctor_info['Location'] = ' '.join(part.text.strip() for part in location_parts) if loc_elem else ""

    # Extracting consultation fee
    fee_elem = doctor_card.find('span', {'data-qa-id': 'consultation_fee'})
    doctor_info['Consultation Fee'] = fee_elem.text.strip() if fee_elem else ""

    return doctor_info

def scrape_doctor_data(location, specialization, max_pages=5):
    """
    Scrapes doctor data from Practo based on the given location and specialization.
    Handles pagination to scrape multiple pages of results.
    
    Args:
        location (str): The city/location to search in.
        specialization (str): The doctor's specialization.
        max_pages (int): Maximum number of pages to scrape.
    
    Returns:
        list: A list of dictionaries containing doctors' data.
    """
    logger.info(f'Starting scrape for {specialization} in {location}')
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }
    
    all_doctors = []
    
    for page in range(1, max_pages + 1):
        url = construct_url(location, specialization, page)
        soup = fetch_page(url, headers)
        
        if not soup:
            logger.warning(f'Skipping page {page} due to fetch errors.')
            continue
        
        doctor_cards = soup.find_all('div', class_="info-section")
        
        if not doctor_cards:
            logger.info(f'No doctor cards found on page {page}. Ending scrape.')
            break  # No more doctors found, exit the loop
        
        for card in doctor_cards:
            doctor_info = parse_doctor_info(card)
            # Append only if all required fields are present
            if all(value != "" for value in doctor_info.values()):
                all_doctors.append(doctor_info)
            else:
                logger.debug('Incomplete doctor info found and skipped.')
        
        logger.info(f'Processed page {page} with {len(doctor_cards)} doctor(s).')
        
        # Polite crawling: wait for a short period before the next request
        time.sleep(1)
    
    logger.info(f'Scraping completed. Total doctors found: {len(all_doctors)}')
    return all_doctors

if __name__ == "__main__":
    # Example usage
    doctors = scrape_doctor_data("Bangalore", "dentist", max_pages=3)
    for doctor in doctors:
        print(doctor)
