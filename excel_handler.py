import os
import pandas as pd
from openpyxl import load_workbook

class ExcelUserHandler:
    def __init__(self, filename='user_login_data.xlsx'):
        self.filename = filename
        self.columns = ['First Name', 'Last Name', 'Email', 'Password']
        
        # Create file if it doesn't exist
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_excel(self.filename, index=False)
    
    def add_user(self, user_data):
        # Read existing data
        try:
            df = pd.read_excel(self.filename)
        except:
            df = pd.DataFrame(columns=self.columns)
        
        # Check if email already exists
        if not df[df['Email'] == user_data['email']].empty:
            return False
        
        # Add new user
        new_row = {
            'First Name': user_data['firstName'],
            'Last Name': user_data['lastName'],
            'Email': user_data['email'],
            'Password': user_data['password']  # Note: In production, store hashed passwords only
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel
        df.to_excel(self.filename, index=False)
        return True
    
    def email_exists(self, email):
        try:
            df = pd.read_excel(self.filename)
            return not df[df['Email'] == email].empty
        except:
            return False