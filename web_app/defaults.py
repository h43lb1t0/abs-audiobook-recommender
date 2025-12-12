"""
This file contains default values for the application.
"""

# Background tasks
BACKGROUND_TASKS = {
    "CHECK_NEW_BOOKS_INTERVAL": 6, # in hours; How often to check for new books
    "CREATE_RECOMMENDATIONS_INTERVAL": 1.2, # in minutes; How often to create recommendations
}


# Recommenderlib constants

RECOMMENDATION_BOOST = {
    "GENRE" : 10, # How much to boost genre matches
    "AUTHOR" : 15, # How much to boost author matches
    "NARRATOR" : 5 # How much to boost narrator matches
}

DEFAULT_RATING = 3 # Default rating for unrated books
MAX_CLUSTERS = 5 # Maximum number of clusters to generate
MOST_COMMON_GENRES = 5 # Maximum number of genres to consider
MOST_COMMON_AUTHORS = 5 # Maximum number of authors to consider
MOST_COMMON_NARRATORS = 5 # Maximum number of narrators to consider

BLEED_ON_NEIGHBOR_PERCENTAGE = 0.1 # Percentage of affinity to bleed to neighbor buckets

# Duration Strictness Logic
DURATION_STRICTNESS_THRESHOLD = 0.02 # 2% - Below this share, user gets a penalty
DURATION_SIGMOID_STEEPNESS = 1.5 # Controls the slope of the sigmoid curve

# Duration Buckets
DURATION_BUCKETS = {
    "super_short": {"max": 3600}, # < 1h
    "short": {"min": 3600, "max": 10800}, # 1h - 3h
    "mid_short": {"min": 10800, "max": 18000}, # 3h - 5h
    "medium": {"min": 18000, "max": 54000}, # 5h - 15h
    "long": {"min": 54000, "max": 86400}, # 15h - 24h
    "epic": {"min": 86400} # > 24h
}
