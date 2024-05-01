import requests
import pathlib
from bs4 import BeautifulSoup
from datetime import datetime

class GithubUtils():
    def __init__(self) -> None:
        pass

    def write_repo_names_to_file(self, topic_url: str = "", filename: str =""):
        """
        Generated with GitHub CoPilot. Somewhat tested (may not handle https exception perfectly but write mode works as designed.)

        Attempts to get a list of repos and their links from a GitHub topic URL and write them to a markdown file in current directory.
        """
        def get_repo_names_and_links(topic_url):
            repo_names_and_links = []
            base_url = 'https://github.com'

            print(f"Getting repositories from {topic_url}...")
            response = requests.get(topic_url)
            try:
                response = requests.get(topic_url)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f'Failed to get response from the URL: {e}')
                return repo_names_and_links

            soup = BeautifulSoup(response.text, 'html.parser')
            repo_elements = soup.find_all('a', class_='Link text-bold wb-break-word')    

            for repo in repo_elements:
                repo_name = repo.text.strip()
                repo_link = base_url + repo['href']
                repo_names_and_links.append((repo_name, repo_link))
                
            return repo_names_and_links

        def write_to_md_file(repos, filename):
            file = pathlib.Path(filename)
            mode = ""
            if file.exists():
                action = input(f'{filename} already exists. Do you want to replace it (y), append to it (a), or cancel (n)? ')
                if action.lower() == 'n':
                    print('Operation cancelled.')
                    return
                elif action.lower() == 'a':
                    print(f'Appending to \"{filename}\".')
                    mode = 'a'
                elif action.lower() == 'y':
                    print(f'Writing to {filename}.')
                    mode = "w"
            else:
                mode = "w"
                print(f'Writing to {filename}.')

            if not mode:
                print("Invalid mode. Operation cancelled...")
                return

            with file.open(mode) as f:
                header = f'source: {topic_url}\n\n\n'
                if mode == "a":
                    header = "\n\n" + header
                f.write(header)
                for name, link in repos:
                    date = datetime.now().strftime('%Y-%m-%d')
                    f.write(f'[{name}]({link}) query_date: {date}\n')
            print('Completed...')

        if not topic_url:
            topic_url: str = input('Enter the topic URL, e.g. https://github.com/topics/deep-anomaly-detection : ')
            if not topic_url:
                print('No URL provided. Exiting...')
                return
        
        if not filename:
            filename: str = input('Enter the filename to write the results to, e.g. repos.md : ')
            if not filename:
                print('No filename provided. Exiting...')
                return
        
        repos = get_repo_names_and_links(topic_url)
        if not repos:
            print(f'No repositories found in {topic_url}. Exiting...')
            return
        
        write_to_md_file(repos, filename)
        
        

