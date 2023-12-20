import json
import requests
from bs4 import BeautifulSoup


def extract_html_content(url):
    """Make an HTTP request to fetch the raw HTML content"""
    try:
        # Make an HTTP GET request to the URL
        headers = {
            'User-Agent': 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        response = requests.get(url, headers=headers)


        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Print the HTML content of the webpage
            print(response.text)
        else:
            # Print an error message if the request was not successful
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    except requests.RequestException as e:
        # Handle any exceptions that may occur during the request
        print(f"Error during the request: {e}")
    
    return response.text


def extract_journal_names(html_content):
    """Parse the HTML content with BeautifulSoup"""

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <a> tags with a specific href pattern
    journal_links = soup.find_all('a', href=lambda href: href and '/journal/' in href)

    # Extract the text content from the links
    journal_names = [link.get_text() for link in journal_links]

    # Print the extracted journal names
    print(f"journal_names: {journal_names}")
    print(f"len(journal_names): {len(journal_names)}")
    return journal_names


def save_journal_names(journal_names):
    """
    Save journal names to json file, using key `journal_names`.

    json file format:
        {
            "journal_names": [
                "Journal of Neuroscience",
                "Neuroscience",
                "Neuron",
                "NeuroImage",
                "Brain Structure and Function",
                "Neurobiology of Aging",
                "Neurobiology of Learning and Memory",
                "Neurobiology of Disea
        }
    """
    with open('journal_names.json', 'w') as f:
        json.dump({"journal_names": journal_names}, f, indent=4)


def main():
    url = "https://research.com/journals-rankings/neuroscience"
    html_content = extract_html_content(url)
    journal_names = extract_journal_names(html_content)
    save_journal_names(journal_names)


if __name__ == "__main__":
    main()
