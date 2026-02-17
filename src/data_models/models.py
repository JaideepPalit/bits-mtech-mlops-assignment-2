from pydantic import BaseModel, Field

class PatientData(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    ca: float
    target: int

    sex_1: bool = Field(..., alias="sex_1.0")
    cp_2: bool = Field(..., alias="cp_2.0")
    cp_3: bool = Field(..., alias="cp_3.0")
    cp_4: bool = Field(..., alias="cp_4.0")
    fbs_1: bool = Field(..., alias="fbs_1.0")
    restecg_1: bool = Field(..., alias="restecg_1.0")
    restecg_2: bool = Field(..., alias="restecg_2.0")
    exang_1: bool = Field(..., alias="exang_1.0")
    slope_2: bool = Field(..., alias="slope_2.0")
    slope_3: bool = Field(..., alias="slope_3.0")
    thal_6: bool = Field(..., alias="thal_6.0")
    thal_7: bool = Field(..., alias="thal_7.0")

    class Config:
        allow_population_by_field_name = True
