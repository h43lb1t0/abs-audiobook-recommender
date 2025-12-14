import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import requests
from db import UserLib, db
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

WRITE_DEBUG_FILES = False

load_dotenv()

ABS_URL = os.getenv("ABS_URL")
ABS_TOKEN = os.getenv("ABS_TOKEN")
ABS_LIB = os.getenv("ABS_LIB")

if not ABS_URL or not ABS_TOKEN:
    logger.error("ABS_URL and ABS_TOKEN must be set in the environment")
    raise ValueError("ABS_URL and ABS_TOKEN must be set in the environment")

if not ABS_LIB:
    logger.warning("ABS_LIB is not set in the environment, using all libraries")

HEADERS = {"Authorization": f"Bearer {ABS_TOKEN}", "Content-Type": "application/json"}


def get_abs_users() -> list:
    """
    Returns all (non admin) users from ABS

    Returns:
        list: A list of users
    """
    users_resp = requests.get(f"{ABS_URL}/api/users", headers=HEADERS)
    users_resp.raise_for_status()
    accounts = users_resp.json().get("users", [])
    abs_users = [account for account in accounts if account.get("type") == "user"]
    clean_users = []
    for abs_user in abs_users:
        user = {
            "id": abs_user.get("id"),
            "username": abs_user.get("username"),
        }
        clean_users.append(user)
    return clean_users


def get_all_items() -> Tuple[dict, dict]:
    """
    Returns all items from ABS

    Returns:
        dict: A map of all items
    """

    items_map = {}
    items_map = {}

    libraries_resp = requests.get(f"{ABS_URL}/api/libraries", headers=HEADERS)
    libraries_resp.raise_for_status()
    libraries = libraries_resp.json().get("libraries", [])

    for lib in libraries:
        if lib.get("mediaType") != "book":
            logger.info(
                f"Skipping library {lib['name']} because it is not a book library"
            )
            continue

        if ABS_LIB and lib["id"] != ABS_LIB:
            logger.info(
                f"Skipping library {lib['name']} because it does not match ABS_LIB"
            )
            continue

        items_url = f"{ABS_URL}/api/libraries/{lib['id']}/items?limit=0&minified=0"
        items_resp = requests.get(items_url, headers=HEADERS)
        items_resp.raise_for_status()

        if WRITE_DEBUG_FILES:
            with Path("items_resp.json").open("w", encoding="utf-8") as f:
                json.dump(items_resp.json(), f, ensure_ascii=False, indent=4)

        for item in items_resp.json().get("results", []):
            metadata = item.get("media", {}).get("metadata", {})

            series_sequence = None
            series_list = metadata.get("series", [])

            if series_list:
                # Prioritize structured series data
                series_name = series_list[0].get("name")
                series_sequence = series_list[0].get("sequence")
            else:
                series_name = metadata.get("seriesName")
                if series_name:
                    # Check for "Name #Sequence" pattern
                    match = re.match(r"^(.*?)\s+#(\d+(?:\.\d+)?)$", series_name)
                    if match:
                        series_name = match.group(1)
                        series_sequence = match.group(2)

            description = metadata.get("description", "")
            if description:
                description = re.sub(r"<[^>]+>", "", description)

            items_map[item["id"]] = {
                "id": item["id"],
                "title": metadata.get("title", item.get("name")),
                "subtitle": metadata.get("subtitle", ""),
                "author": metadata.get("authorName", "Unknown"),
                "narrator": metadata.get("narratorName", "Unknown"),
                "series": series_name,
                "series_sequence": series_sequence,
                "genres": metadata.get("genres", []),
                "tags": item.get("media", {}).get("tags", []),
                "cover": item.get("media", {}).get("coverPath"),
                "description": description,  # Fetch description
                "lib_name": lib["name"],  # Useful for debugging or filtering
                "duration_seconds": item.get("media", {}).get("duration"),
            }

            if WRITE_DEBUG_FILES:
                with Path("items_map.json").open("w", encoding="utf-8") as f:
                    json.dump(items_map, f, ensure_ascii=False, indent=4)

            if (
                not items_map[item["id"]]["author"]
                or items_map[item["id"]]["author"] == "Unknown"
            ):
                authors = metadata.get("authors", [])
                if authors:
                    items_map[item["id"]]["author"] = authors[0].get("name")

    return items_map


def get_finished_books(items_map: dict, user_id: str = None) -> Tuple[set, set, set]:
    """
    Returns the finished books from ABS (books count as finished if they are finished or if they are 97% read)

    Also saves/updates the progress to the local UserLib table.

    Args:
        items_map (dict): The items map
        user_id (str): The user ID to fetch finished books for. If None, uses the token's user (api/me).

    Returns:
        Tuple[set, dict, set]: A tuple containing the finished books (set), in-progress books (dict: id->progress), and the finished keys (set)
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

    media_progress = user_data.get("mediaProgress", [])

    finished_ids = set()
    in_progress_ids = {}  # Changed to dict to store progress

    finished_keys = set()

    # Identify the user ID to save to DB
    current_user_id = user_id
    if not current_user_id:
        current_user_id = user_data.get("id")

    # Prepare for DB update if we have a user ID and are in an app context
    should_sync_db = False

    if current_user_id:
        try:
            # Check if we are in an app context (db.session works)
            from flask import current_app

            if current_app:
                should_sync_db = True
        except ImportError:
            pass
        except RuntimeError:
            # working outside of application context
            pass

    user_lib_map = {}
    if should_sync_db:
        try:
            existing_entries = UserLib.query.filter_by(user_id=current_user_id).all()
            user_lib_map = {entry.book_id: entry for entry in existing_entries}
        except Exception as e:
            logger.error(f"Error fetching UserLib entries for syncing: {e}")
            should_sync_db = False

    # First pass: Determine the final status for each item, handling duplicates
    final_status_map = {}  # item_id -> status

    for mp in media_progress:
        item_id = mp.get("libraryItemId")

        is_finished = mp.get("isFinished", False)
        progress = mp.get("progress", 0.0)

        currentTime = mp.get("currentTime", 0.0)

        status = None

        if is_finished or progress >= 0.97:
            # Also add to return sets
            finished_ids.add(item_id)
            status = "finished"
            if item_id in items_map:
                book = items_map[item_id]
                finished_keys.add((book["title"], book["author"]))

        elif progress > 0 or currentTime > 0:
            in_progress_ids[item_id] = progress
            status = "reading"

        if status:
            if item_id in final_status_map:
                # Conflict resolution: finished > reading
                if final_status_map[item_id] == "reading" and status == "finished":
                    final_status_map[item_id] = "finished"
            else:
                final_status_map[item_id] = status

    # Sync to DB
    if should_sync_db:
        for item_id, status in final_status_map.items():
            try:
                if item_id in user_lib_map:
                    entry = user_lib_map[item_id]
                    if entry.status != status:
                        # Don't overwrite abandoned status with reading status
                        if entry.status == "abandoned" and status == "reading":
                            logger.debug(
                                f"Skipping status update for item {item_id}: keeping 'abandoned' despite 'reading' status from ABS"
                            )
                            continue

                        logger.debug(
                            f"Updating status for item {item_id} from {entry.status} to {status}"
                        )
                        entry.status = status
                        entry.updated_at = datetime.now().isoformat()
                else:
                    new_entry = UserLib(
                        user_id=current_user_id,
                        book_id=item_id,
                        status=status,
                        rating=None,
                        updated_at=datetime.now().isoformat(),
                    )
                    db.session.add(new_entry)
                    user_lib_map[item_id] = new_entry  # Update local map
            except Exception as e:
                logger.error(f"Error updating UserLib for item {item_id}: {e}")

    if should_sync_db:
        try:
            db.session.commit()
            logger.info(f"Synced UserLib progress for user {current_user_id}")
        except Exception as e:
            logger.error(f"Error committing UserLib changes: {e}")
            db.session.rollback()

    return finished_ids, in_progress_ids, finished_keys
