import os

from dotenv import load_dotenv
from huggingface_hub import login


package_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(package_dir)
package_dir = os.path.dirname(package_dir)
env_path = os.path.join(package_dir, ".env")

if not os.environ.get("QFLUX_DOTENV_LOADED"):
    load_dotenv(env_path)
    os.environ["QFLUX_DOTENV_LOADED"] = "1"
    print("Environment variables loaded from .env file")
    if "HF_TOKEN" in os.environ:
        try:
            login(token=os.environ["HF_TOKEN"])
            print("Logged in to Hugging Face")
        except Exception as e:
            print(f"Warning: Failed to login to Hugging Face: {e}")
    else:
        print("HF_TOKEN not found in environment variables. Skipping Hugging Face login.")
