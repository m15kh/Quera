from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

log_pipeline = make_pipeline(
    SimpleImputer(strategy='mediam')
    
    
)


preprocessing = ColumnTransformer([
    ('log', log_pipeline, [''])
    
])