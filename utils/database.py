import pandas as pd
import os
from datetime import datetime

class AttendanceDB:
    def __init__(self, attendance_file="data/attendance.csv"):
        self.attendance_file = attendance_file
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Student_ID', 'Department', 'Date', 'Time', 'Status'])
            df.to_csv(self.attendance_file, index=False)
    
    def mark_attendance(self, name, student_details):
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Read existing data
        df = pd.read_csv(self.attendance_file)
        
        # Check if already marked today
        existing = df[(df['Name'] == name) & (df['Date'] == current_date)]
        
        if len(existing) == 0:
            new_record = pd.DataFrame([{
                'Name': name,
                'Student_ID': student_details.get('student_id', 'N/A'),
                'Department': student_details.get('department', 'N/A'),
                'Date': current_date,
                'Time': current_time,
                'Status': 'Present'
            }])
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(self.attendance_file, index=False)
            return True
        return False
    
    def get_attendance_records(self):
        if not os.path.exists(self.attendance_file):
            return pd.DataFrame(columns=['Name', 'Student_ID', 'Department', 'Date', 'Time', 'Status'])
        return pd.read_csv(self.attendance_file)
    
    def get_statistics(self):
        df = self.get_attendance_records()
        if len(df) == 0:
            return {
                'total_records': 0,
                'unique_students': 0,
                'unique_days': 0,
                'today_count': 0,
                'average_per_day': 0
            }
        
        return {
            'total_records': len(df),
            'unique_students': df['Name'].nunique(),
            'unique_days': df['Date'].nunique(),
            'today_count': len(df[df['Date'] == datetime.now().strftime("%Y-%m-%d")]),
            'average_per_day': len(df) / max(df['Date'].nunique(), 1)
        }