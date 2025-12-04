import os
import requests
import json
import logging
from dotenv import load_dotenv
import re
from typing import Tuple


logger = logging.getLogger(__name__)

load_dotenv()

ABS_URL = os.getenv("ABS_URL")
ABS_TOKEN = os.getenv("ABS_TOKEN")
ABS_LIB = os.getenv("ABS_LIB")

if not ABS_URL or not ABS_TOKEN:
    logger.error("ABS_URL and ABS_TOKEN must be set in the environment")
    raise ValueError("ABS_URL and ABS_TOKEN must be set in the environment")

if not ABS_LIB:
    logger.warning("ABS_LIB is not set in the environment, using all libraries")

HEADERS = {
    "Authorization": f"Bearer {ABS_TOKEN}",
    "Content-Type": "application/json"
}


def get_abs_users() -> list:
    """
    Returns all (non admin) users from ABS

    Returns:
        list: A list of users
    """
    users_resp = requests.get(f"{ABS_URL}/api/users", headers=HEADERS)
    users_resp.raise_for_status()
    accounts = users_resp.json().get('users', [])
    abs_users = [account for account in accounts if account.get('type') == 'user']
    clean_users = []
    for abs_user in abs_users:
        user = {
            'id': abs_user.get('id'),
            'username': abs_user.get('username'),
        }
        clean_users.append(user)
    return clean_users
        



def get_all_items() -> Tuple[dict, dict]:
    """
    Returns all items from ABS

    A series with more than 10 books is filtered out

    Returns:
        Tuple[dict, dict]: A tuple containing the items map and series counts
    """ 

    items_map = {}
    series_counts = {}

    libraries_resp = requests.get(f"{ABS_URL}/api/libraries", headers=HEADERS)
    libraries_resp.raise_for_status()
    libraries = libraries_resp.json().get('libraries', [])

    for lib in libraries:
        if lib.get('mediaType') != 'book':
            logger.info(f"Skipping library {lib['name']} because it is not a book library")
            continue

        if ABS_LIB and lib['id'] != ABS_LIB:
            logger.info(f"Skipping library {lib['name']} because it does not match ABS_LIB")
            continue

        items_url = f"{ABS_URL}/api/libraries/{lib['id']}/items?limit=0&minified=0"
        items_resp = requests.get(items_url, headers=HEADERS)
        items_resp.raise_for_status()

        for item in items_resp.json().get('results', []):
            metadata = item.get('media', {}).get('metadata', {})

            series_sequence = None
            series_list = metadata.get('series', [])
            
            if series_list:
                # Prioritize structured series data
                series_name = series_list[0].get('name')
                series_sequence = series_list[0].get('sequence')
            else:
                series_name = metadata.get('seriesName')
                if series_name:
                    # Check for "Name #Sequence" pattern
                    match = re.match(r'^(.*?)\s+#(\d+(?:\.\d+)?)$', series_name)
                    if match:
                        series_name = match.group(1)
                        series_sequence = match.group(2)


            if series_name:
                series_counts[series_name] = series_counts.get(series_name, 0) + 1

            items_map[item['id']] = {
                'id': item['id'],
                'title': metadata.get('title', item.get('name')),
                'author': metadata.get('authorName', 'Unknown'),
                'series': series_name,
                'series_sequence': series_sequence,
                'genres': metadata.get('genres', []),
                'cover': item.get('media', {}).get('coverPath'),
                'lib_name': lib['name'] # Useful for debugging or filtering
            }

            if not items_map[item['id']]['author'] or items_map[item['id']]['author'] == 'Unknown':
                authors = metadata.get('authors', [])
                if authors:
                    items_map[item['id']]['author'] = authors[0].get('name')

    return items_map, series_counts 



def get_finished_books(items_map: dict, user_id: str = None) -> Tuple[set, set, set]:
    """
    Returns the finished books from ABS (books count as finished if they are finished or if they are 97% read)

    Args:
        items_map (dict): The items map
        user_id (str): The user ID to fetch finished books for. If None, uses the token's user (api/me).

    Returns:
        Tuple[set, set, set]: A tuple containing the finished books, in-progress books, and the finished keys
    """

    if user_id:
        url = f"{ABS_URL}/api/users/{user_id}"
    else:
        url = f"{ABS_URL}/api/me"

    logger.info(f"Fetching finished books from: {url}")

    try:
        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        user_data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user data: {e}")
        return set(), set(), set()

    # The user object structure is the same for /api/me and /api/users/{id}
    # However, /api/users/{id} might return the user object directly or wrapped.
    # Based on get_abs_users, /api/users returns { users: [...] }
    # Usually /api/users/{id} returns the user object directly.
    
    # Let's assume it returns the user object directly which contains mediaProgress.
    media_progress = user_data.get('mediaProgress', [])
    
    finished_ids = set()
    in_progress_ids = set()

    finished_keys = set()

    for mp in media_progress:
        item_id = mp.get('libraryItemId')

        is_finished = mp.get('isFinished', False)
        progress = mp.get('progress', 0.0)

        currentTime = mp.get('currentTime', 0.0)

        if is_finished or progress >= 0.97:
            finished_ids.add(item_id)
            if item_id in items_map:
                book = items_map[item_id]
                finished_keys.add((book['title'], book['author']))
                
        elif progress > 0 or currentTime > 0:
            in_progress_ids.add(item_id)

    return finished_ids, in_progress_ids, finished_keys

