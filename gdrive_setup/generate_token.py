import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

def generate_token():
    """Generate and save OAuth token for Google Drive API"""
    creds = None

    # Check if token already exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

        if creds and creds.valid:
            print("Valid token already exists!")
            print("Token file: token.pickle")
            return True
        elif creds and creds.expired and creds.refresh_token:
            print("Token expired, refreshing...")
            try:
                creds.refresh(Request())
                print("Token refreshed successfully!")
            except Exception as e:
                print(f"Failed to refresh token: {e}")
                print("Generating new token...")
                creds = None

    # Generate new token if needed
    if not creds or not creds.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            print("New token generated successfully!")
        except FileNotFoundError:
            print("Error: 'credentials.json' not found!")
            print("Please download OAuth 2.0 Client ID credentials from Google Cloud Console")
            return False
        except Exception as e:
            print(f"Error generating token: {e}")
            return False

    # Save the credentials for next run
    try:
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
        print("Token saved to 'token.pickle'")
        return True
    except Exception as e:
        print(f"Error saving token: {e}")
        return False

def check_token_status():
    """Check current token status"""
    if not os.path.exists('token.pickle'):
        print("No token file found. Run generate_token() first.")
        return

    try:
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

        if creds.valid:
            print("✅ Token is valid and ready to use")
        elif creds.expired:
            if creds.refresh_token:
                print("⚠️  Token expired but can be refreshed")
            else:
                print("❌ Token expired and cannot be refreshed")
        else:
            print("❌ Token is invalid")

    except Exception as e:
        print(f"Error reading token: {e}")

if __name__ == "__main__":
    print("=== Google Drive Token Generator ===")
    print()

    # Check current status
    check_token_status()
    print()

    # Generate or refresh token
    if generate_token():
        print()
        print("✅ Success! Token is ready.")
    else:
        print()
        print("❌ Failed to generate token.")
        print("Please check your credentials.json file and try again.")