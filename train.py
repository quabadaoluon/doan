import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_and_save():
    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv('vehicles.csv')

    # 2. Tiền xử lý
    features = ['price', 'year', 'manufacturer', 'condition', 'fuel',
                'odometer', 'title_status', 'transmission', 'drive',
                'type', 'paint_color', 'lat', 'long']

    df = df[features].dropna()
    df = df[(df['price'] > 1000) & (df['price'] < 60000)]
    df = df[(df['odometer'] > 0) & (df['odometer'] < 300000)]

    # Trích xuất danh sách các giá trị duy nhất cho giao diện Streamlit
    cat_cols = ['manufacturer', 'condition', 'fuel', 'title_status',
                'transmission', 'drive', 'type', 'paint_color']
    unique_values = {col: sorted(df[col].unique().tolist()) for col in cat_cols}

    y = np.log1p(df['price'])
    X = df.drop(columns=['price'])

    # 3. Định nghĩa Pipelines
    numeric_features = ['year', 'odometer', 'lat', 'long']
    categorical_features = cat_cols

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('ridge', Ridge(alpha=10))
    ])

    # 4. Huấn luyện
    print("Đang huấn luyện mô hình (bước này có thể mất vài phút)...")
    model_pipeline.fit(X, y)

    save_data = {
        'model': model_pipeline,
        'categories': unique_values
    }

    joblib.dump(save_data, 'ridge_car_model.pkl')
    print("--- THÀNH CÔNG ---")
    print("Đã tạo file ridge_car_model.pkl tương thích 100% với máy của bạn!")


if __name__ == "__main__":
    train_and_save()