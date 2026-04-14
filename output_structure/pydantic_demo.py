from typing import Optional

from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str
    age: int
    city: Optional[str] = None
    branch: str
    email: EmailStr
    cgpa : float = Field(default = 0.0, gt = 0.0, lt = 10.0)
    
student = Student(name="John", age="20",branch="Computer Science",email="john@example.com",cgpa=1.5,description="John is a student of Computer Science")
student_dict = dict(student)
print(student.model_dump_json())
print(student_dict)