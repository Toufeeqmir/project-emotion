import requests
import os

# ==========================================
# 1. SETUP
# ==========================================
# The Live URL of your deployed model
API_URL = "https://toufeeq04-emotion-backend.hf.space/predict"

# PASTE YOUR PATH BELOW (Keep the 'r' at the start!)
# Example: r"C:\Users\You\Downloads\my_image.jpg"
TEST_IMAGE_PATH = r"E:\Aicoach\python\New-dataset\test\happy\aug_28339ffhq_1015.png" 

print(f"--- Configuration ---")
print(f"Target URL: {API_URL}")
print(f"Image Path: {TEST_IMAGE_PATH}")

# ==========================================
# 2. SAFETY CHECK (Verifies file exists)
# ==========================================
if not os.path.exists(TEST_IMAGE_PATH):
    print("\n❌ CRITICAL ERROR: File not found!")
    print(f"The computer looked for: {TEST_IMAGE_PATH}")
    print("Please check:\n1. Did you paste the full path?\n2. Did you keep the 'r' before the quote?\n3. Does the file actually exist?")
    exit()

print("\n✅ File found! Sending to cloud...")

# ==========================================
# 3. SEND REQUEST
# ==========================================
try:
    with open(TEST_IMAGE_PATH, "rb") as f:
        # Send the file to your Render API
        response = requests.post(API_URL, files={"file": f})
    
    # Check if the server is happy (200 OK)
    if response.status_code == 200:
        print("\n🎉 SUCCESS! Server responded:")
        print(response.json())
    else:
        print(f"\n⚠️ SERVER ERROR (Status {response.status_code}):")
        print(response.text)

except Exception as e:
    print(f"\n❌ CONNECTION ERROR: {e}")