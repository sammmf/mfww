## modules/dropbox_integration.py

import dropbox
import streamlit as st

def initialize_dropbox():
    """Initialize Dropbox client using API key from Streamlit secrets."""
    try:
        DROPBOX_API_KEY = st.secrets["DROPBOX_API_KEY"]
        dbx = dropbox.Dropbox(DROPBOX_API_KEY)
        return dbx
    except KeyError:
        st.error("Dropbox API key is missing. Please add it to Streamlit secrets.")
        return None
    except Exception as e:
        st.error(f"Failed to connect to Dropbox: {e}")
        return None

def download_data_file(dbx):
    """Download the Excel file from Dropbox, trying multiple possible paths."""
    possible_paths = [
        "/work/mccall_farms/mccall_shared_data/daily_data.xlsx",
        "/Work/McCall_Farms/McCall_Shared_Data/daily_data.xlsx"
    ]

    for path in possible_paths:
        try:
            metadata, res = dbx.files_download(path)
            with open("daily_data.xlsx", "wb") as f:
                f.write(res.content)
            print(metadata)  # For debugging
            return True
        except dropbox.exceptions.ApiError as e:
            st.warning(f"Failed to download file from: {path}. Trying next path...")
            print(f"Error: {e}")
    st.error("Unable to download the Excel file from Dropbox. Please check the paths.")
    return False
