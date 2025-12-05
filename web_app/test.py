import os
import dotenv

dotenv.load_dotenv()

use_gemini = os.getenv("USE_GEMINI")

def bool_from_env(var):
    return var.lower() == 'true'

USE_GEMINI = bool_from_env(use_gemini)


print(USE_GEMINI)