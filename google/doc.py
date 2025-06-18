from googleapiclient.discovery import build
from google.credentials import get_creds

creds = get_creds()

def extract_google_doc(doc_id):
    docs_service = build('docs', 'v1', credentials=creds)
    doc = docs_service.documents().get(documentId=doc_id).execute()
    text = "".join([el.get("paragraph", {}).get("elements", [{}])[0].get("textRun", {}).get("content", "")
                    for el in doc.get("body", {}).get("content", [])])
    return text