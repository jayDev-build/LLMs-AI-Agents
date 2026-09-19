from pydantic import BaseModel, Field, EmailStr
from typing import Optional
class Student(BaseModel):
    name: str = 'Yashit'
    age: Optional[int] = None
    grade: str
    email: Optional[EmailStr] = Field(None, description="The student's email address")
    cgpa: float = Field(..., description="The student's CGPA", ge=0.0, le=10.0)

student = {
    'name': 'John Doe',
    'age': '20',
    'grade': 'A',
    'email': 'abc@gmail.com',
    'cgpa': 9.5

}

student_obj = Student(**student)

print(student_obj)