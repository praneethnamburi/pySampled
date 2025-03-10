import urllib.request
import os

def main():
    airpls_url = "https://raw.githubusercontent.com/zmzhang/airPLS/master/airPLS.py"
    destination = os.path.join(os.path.dirname(__file__), "..", "pysampled", "airPLS.py")

    try:
        print(f"Downloading airPLS.py from {airpls_url}")
        urllib.request.urlretrieve(airpls_url, destination)
        print("airPLS.py successfully downloaded.")
    except Exception as e:
        print(f"Failed to download airPLS.py: {e}")

if __name__ == "__main__":
    main()
