from tarsDAO import AIDAO

# MODULE: upload.py
# LAST UPDATED: 03/25/2023
# AUTHOR: MATTHEW MAR
# FUNCTION : Upload CELESTE' memory to database


# Function to upload CELESTE memory to database
# Using tarsDAO
# RETURNS: boolean for success, failure
def main():
    # Print a message to indicate that the CELESTE memory is being uploaded
    print("INITIALIZING DATABASE CONNECTION...")
    # Create an instance of the AIDAO class
    aidao = AIDAO()
    print("UPLOADING CELESTE MEMORY")
    # Call the uploadTARS method to upload the CELESTE memory
    uploaded = aidao.uploadTARS()
    # Check if the upload was successful and print a message accordingly
    if uploaded:
        print("CELESTE MEMORY SUCCESSFULLY UPLOADED")
    else:
        print("CELESTE MEMORY FAILED TO UPLOAD..")
    return uploaded


# If the script is being run directly, call the main function
if __name__ == "__main__":
    main()
