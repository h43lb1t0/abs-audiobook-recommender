"""
This file contains default values for the application.
"""

# Background tasks
BACKGROUND_TASKS = {
    "CHECK_NEW_BOOKS_INTERVAL": 5, # in minutes; How often to check for new books
    "CREATE_RECOMMENDATIONS_INTERVAL": 2, # in minutes; How often to create recommendations
}


# Recommenderlib constants

RECOMMENDATION_BOOST = {
    "GENRE" : 10, # How much to boost genre matches
    "AUTHOR" : 15 # How much to boost author matches
}

DEFAULT_RATING = 3 # Default rating for unrated books
MAX_CLUSTERS = 5 # Maximum number of clusters to generate
MOST_COMMON_GENRES = 5 # Maximum number of genres to consider
MOST_COMMON_AUTHORS = 5 # Maximum number of authors to consider
