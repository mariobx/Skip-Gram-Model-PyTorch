import pandas as pd
import sys
import os

dataframe = pd.read_csv(os.path.join('/home/mariom/Work/Fordham/labs/Fall24/final/', 'processed-data-11-25-24.csv'))

def create_student_term_course(df: pd.DataFrame, s_id: str, semester_col: str, dept_col: str) -> pd.DataFrame:
    """
    Groups a dataframe by student ID and returns their courses as a list,
    ordered by the semester (newest to oldest).
    
    Args:
        df (pd.DataFrame): The input dataframe containing student, course, and semester information.
        s_id (str): Column name for student IDs.
        semester_col (str): Column name for semesters.
        
    Returns:
        pd.DataFrame: A dataframe with student IDs and their ordered course lists.
    """
    # Ensure the courses are ordered by semester
    ordered_df = df.sort_values(by=[s_id, semester_col, dept_col], ascending=[True, True, True])  # Newest to oldest
    
    # Aggregate the ordered courses by student
    agg_df = (
        ordered_df.groupby([s_id])
        .agg({'CourseTitle': lambda x: [course for course in x if pd.notna(course)]})  # Remove NaN here
        .reset_index()
    )
    
    agg_df = agg_df.rename(columns={'CourseTitle': 'CourseSequence'})
    
    agg_df = agg_df[agg_df['CourseSequence'].apply(len) >= 6]
    
    agg_df = agg_df.reset_index(drop=True)
    
    agg_df.to_csv('agg.csv', index=False)
    
    return agg_df


sav_df = create_student_term_course(dataframe, 'SID', 'Semester', 'Department')
sav_df.to_csv(os.path.join('/home/mariom/Work/Fordham/labs/Fall24/final/round2pytorchboogaloo/', 'retgre.csv'), index=False)