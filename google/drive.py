from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.credentials import get_creds

creds = get_creds()

def get_google_drive_files():
  """Shows basic usage of the Drive v3 API.
  Prints the names and ids of the first 10 files the user has access to.
  """
  try:
    service = build("drive", "v3", credentials=creds)
    # Call the Drive v3 API
    results = (
        service.files()
        .list(
          q="mimeType!='application/pdf' and"
              "(name contains 'Closer Pets Hub Firmware Specification 2022-11-18'" 
                  # "or name contains 'IoT Messaging System Draft 20'"
                  "or name contains 'Closer-pet Functional documentation v1.2')",
          fields="nextPageToken, files(id, name)"
        )
        .execute()
    )
    items = results.get("files", [])
    if not items:
      print("No files found.")
      return None
    return items
  except HttpError as error:
    # TODO(developer) - Handle errors from drive API.
    print(f"An error occurred: {error}")