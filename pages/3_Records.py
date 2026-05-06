import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Attendance Records", page_icon="📊")

st.title("📊 Attendance Records")
st.markdown("### View and analyze attendance history")

from utils.database import AttendanceDB
from utils.optimized_face import OptimizedFaceRecognizer

attendance_db = AttendanceDB()
recognizer = OptimizedFaceRecognizer()

df = attendance_db.get_attendance_records()

if len(df) == 0:
    st.info("📭 No attendance records found yet")
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🔍 Filter Options")

# Date filter
date_options = ["All Time", "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "This Month", "Last Month"]
date_filter = st.sidebar.selectbox("Date Range", date_options)

# Department filter
departments = ["All"] + sorted(df['Department'].unique().tolist())
dept_filter = st.sidebar.selectbox("Department", departments)

# Student filter
students = ["All"] + sorted(df['Name'].unique().tolist())
student_filter = st.sidebar.selectbox("Student", students)

# Apply filters
filtered_df = df.copy()

# Date filter
if date_filter == "Today":
    filtered_df = filtered_df[filtered_df['Date'] == datetime.now().strftime("%Y-%m-%d")]
elif date_filter == "Yesterday":
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    filtered_df = filtered_df[filtered_df['Date'] == yesterday]
elif date_filter == "Last 7 Days":
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    filtered_df = filtered_df[filtered_df['Date'] >= week_ago]
elif date_filter == "Last 30 Days":
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    filtered_df = filtered_df[filtered_df['Date'] >= month_ago]
elif date_filter == "This Month":
    filtered_df = filtered_df[filtered_df['Date'].str.startswith(datetime.now().strftime("%Y-%m"))]
elif date_filter == "Last Month":
    last_month = datetime.now().replace(day=1) - timedelta(days=1)
    filtered_df = filtered_df[filtered_df['Date'].str.startswith(last_month.strftime("%Y-%m"))]

# Department filter
if dept_filter != "All":
    filtered_df = filtered_df[filtered_df['Department'] == dept_filter]

# Student filter
if student_filter != "All":
    filtered_df = filtered_df[filtered_df['Name'] == student_filter]

# Display metrics
if len(filtered_df) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", len(filtered_df))
    with col2:
        st.metric("👥 Unique Students", filtered_df['Name'].nunique())
    with col3:
        st.metric("📅 Active Days", filtered_df['Date'].nunique())
    with col4:
        avg = len(filtered_df) / max(filtered_df['Date'].nunique(), 1)
        st.metric("📈 Average/Day", f"{avg:.1f}")
    
    # Charts Section
    st.markdown("---")
    st.markdown("### 📈 Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Charts", "👥 Student Analysis", "📋 Detailed Records"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily attendance trend
            daily_attendance = filtered_df.groupby('Date').size().reset_index(name='Count')
            fig = px.line(daily_attendance, x='Date', y='Count', 
                         title='Daily Attendance Trend',
                         markers=True,
                         line_shape='linear')
            fig.update_traces(line_color='#4CAF50', line_width=3, marker_size=8)
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Department distribution
            dept_counts = filtered_df['Department'].value_counts()
            fig = px.pie(values=dept_counts.values, names=dept_counts.index,
                        title='Attendance by Department',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top students
        st.markdown("### 🏆 Top Performing Students")
        top_students = filtered_df['Name'].value_counts().head(10)
        fig = px.bar(x=top_students.values, y=top_students.index, 
                     orientation='h',
                     title='Top 10 Students by Attendance',
                     color=top_students.values,
                     color_continuous_scale='Viridis',
                     text=top_students.values)
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400, xaxis_title='Number of Days Present')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Student-wise attendance percentage
        total_days = filtered_df['Date'].nunique()
        student_summary = filtered_df.groupby(['Name', 'Student_ID', 'Department']).size().reset_index(name='Present Days')
        student_summary['Percentage'] = (student_summary['Present Days'] / total_days * 100).round(2)
        student_summary = student_summary.sort_values('Percentage', ascending=False)
        
        st.markdown("### 📊 Student Attendance Summary")
        st.dataframe(student_summary, use_container_width=True, hide_index=True)
        
        # Individual student report
        st.markdown("### 🔍 Individual Student Report")
        selected_student = st.selectbox("Select Student", sorted(filtered_df['Name'].unique()))
        
        if selected_student:
            student_data = filtered_df[filtered_df['Name'] == selected_student]
            student_info = recognizer.student_details.get(selected_student, {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Present", len(student_data))
            with col2:
                student_percentage = (len(student_data) / total_days * 100) if total_days > 0 else 0
                st.metric("Attendance Percentage", f"{student_percentage:.1f}%")
            with col3:
                st.metric("Student ID", student_info.get('student_id', 'N/A'))
            
            # Student attendance timeline
            student_data['Date'] = pd.to_datetime(student_data['Date'])
            student_data = student_data.sort_values('Date')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=student_data['Date'], y=student_data['Time'],
                                     mode='markers+lines',
                                     name='Attendance',
                                     marker=dict(size=10, color='#4CAF50'),
                                     line=dict(color='#4CAF50', width=2)))
            fig.update_layout(title=f'Attendance Timeline for {selected_student}',
                             xaxis_title='Date',
                             yaxis_title='Time',
                             height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Attendance dates list
            st.markdown("**Attendance Dates:**")
            dates = student_data['Date'].dt.strftime('%Y-%m-%d').tolist()
            st.write(", ".join(dates))
    
    with tab3:
        # Detailed records table
        st.markdown("### 📋 Detailed Attendance Records")
        display_df = filtered_df[['Name', 'Student_ID', 'Department', 'Date', 'Time']].sort_values('Date', ascending=False)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Search functionality
        st.markdown("### 🔍 Search Records")
        search_term = st.text_input("Search by Name or Student ID", placeholder="Enter name or ID...")
        if search_term:
            search_results = filtered_df[filtered_df['Name'].str.contains(search_term, case=False) | 
                                        filtered_df['Student_ID'].str.contains(search_term, case=False)]
            if len(search_results) > 0:
                st.dataframe(search_results[['Name', 'Student_ID', 'Department', 'Date', 'Time']], 
                            use_container_width=True)
            else:
                st.info("No matching records found")
    
    # Export section
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, 
                          f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                          "text/csv",
                          use_container_width=True)
    
    with col2:
        # Prepare summary report
        summary_data = {
            'Report Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Date Range': date_filter,
            'Total Records': len(filtered_df),
            'Unique Students': filtered_df['Name'].nunique(),
            'Active Days': filtered_df['Date'].nunique(),
            'Average Daily Attendance': f"{len(filtered_df) / max(filtered_df['Date'].nunique(), 1):.1f}"
        }
        summary_df = pd.DataFrame([summary_data])
        summary_csv = summary_df.to_csv(index=False)
        st.download_button("📊 Download Summary Report", summary_csv,
                          f"summary_report_{datetime.now().strftime('%Y%m%d')}.csv",
                          "text/csv",
                          use_container_width=True)

else:
    st.warning("⚠️ No records found for the selected filters")

# Back button
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("app.py")