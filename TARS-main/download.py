from tarsDAO import AIDAO

# MODULE: download.py
# LAST UPDATED: 03/25/2023
# AUTHOR: MATTHEW MAR
# FUNCTION : Upload CELESTE' memory to database

# Function to download CELESTE memory to database
# Using tarsDAO
# RETURNS: boolean for success, failure


def main():
    # Print a message to indicate that the CELESTE memory is being downloaded
    print("INITIALIZING DATABASE CONNECTION...")
    # Instantiate AIDAO class
    aidao = AIDAO()
    print("DOWNLOADING CELESTE MEMORY")
    # Make a call to the downloadTARS method to download CELESTE's memory
    downloaded = aidao.downloadTARS()
    # Check if CELESTE'S memory was downloaded
    if downloaded:
        print("CELESTE MEMORY SUCCESSFULLY DOWNLOADED")
    else:
        print("CELESTE MEMORY FAILED TO DOWNLOAD..")
    return downloaded


# If the script is being run directly, call the main function
if __name__ == "__main__":
    main()
