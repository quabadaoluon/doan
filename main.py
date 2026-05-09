import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


st.set_page_config(
    page_title="CarValue AI | Professional",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* 1. Cấu hình chung */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 18px;
    }

    .stApp {
        background: #0f172a;
        color: #f1f5f9;
    }

    /* Giới hạn độ rộng của trang để các ô không bị quá dài */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1000px; /* Thu hẹp lại để các ô nhập liệu tự động ngắn lại một cách tự nhiên */
    }

    /* 2. Tiêu đề */
    .header-box {
        text-align: center;
        padding-bottom: 30px;
    }

    .main-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 3. Note Box */
    .note-box {
        background: #172554;
        border-left: 5px solid #38bdf8;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 17px;
    }

    /* 4. CHỈNH THẲNG HÀNG VÀ KÍCH THƯỚC CHỮ */
    /* Ép tất cả các nhãn (labels) có cùng chiều cao để input bên dưới thẳng hàng ngang */
    [data-testid="stWidgetLabel"] p {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        min-height: 50px; /* Chiều cao cố định cho label */
        display: flex;
        align-items: flex-end;
    }

    /* Tăng kích thước chữ bên trong các ô nhập liệu */
    .stSelectbox div[data-baseweb="select"], 
    .stTextInput input, 
    .stNumberInput input {
        font-size: 18px !important;
        height: 50px !important;
        border-radius: 8px !important;
    }

    /* Fix lỗi lệch khung: Đảm bảo container của widget không bị tràn */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-end; /* Căn chỉnh các cột theo đáy */
    }

    /* 5. Nút bấm */
    div.stButton > button {
        width: 100%;
        background: #2563eb;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 800;
        margin-top: 20px;
    }

    /* 6. Kết quả */
    .result-container {
        background: #0ea5e9;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: white;
    }

    .result-container h2 {
        font-size: 50px !important;
        font-weight: 900;
        margin: 0;
    }

    /* 7. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 700;
        padding: 10px 25px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_saved_data():
    try:
        data = joblib.load("ridge_car_model.pkl")
        return data["model"], data["categories"]
    except Exception:
        return None, None


model, categories = load_saved_data()
geolocator = Nominatim(user_agent="car_value_ai")


st.markdown("""
<div class="header-box">
    <h1 class="main-title">
        PHẦN MỀM DỰ ĐOÁN GIÁ XE CŨ
    </h1>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
❌ Không tìm thấy file model.

Vui lòng kiểm tra:
- File ridge_car_model.pkl có tồn tại không
- File có nằm cùng thư mục với app.py không
- File model có bị lỗi hoặc sai định dạng không
""")
    st.stop()

tab_single, tab_batch = st.tabs([
    "🎯 DỰ ĐOÁN ĐƠN LẺ",
    "📊 XỬ LÝ DỮ LIỆU LỚN"
])

with tab_single:

    st.markdown("""
    <div class="note-box">
        <b>Lưu ý:</b><br>
        • Nhập vị trí chính xác để hệ thống định vị tốt hơn<br>
        • ODO nên nhập theo đơn vị kilomet (km)<br>
        • Kết quả chỉ mang tính tham khảo theo thị trường
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        manufacturer = st.selectbox(
            "Hãng xe",
            categories["manufacturer"]
        )

        year = st.slider(
            "Năm sản xuất",
            1990,
            2023,
            2018
        )

        fuel = st.selectbox(
            "Nhiên liệu",
            categories["fuel"]
        )

    with c2:
        type_car = st.selectbox(
            "Kiểu dáng",
            categories["type"]
        )

        odometer = st.number_input(
            "Số ODO (km)",
            value=30000
        )

        transmission = st.selectbox(
            "Hộp số",
            categories["transmission"]
        )

    with c3:
        condition = st.selectbox(
            "Tình trạng",
            categories["condition"]
        )

        drive = st.selectbox(
            "Hệ dẫn động",
            categories["drive"]
        )

        location_input = st.text_input(
            "Vị trí niêm yết",
            value="California"
        )

    sub1, sub2 = st.columns(2)

    with sub1:
        paint_color = st.selectbox(
            "Màu sắc ngoại thất",
            categories["paint_color"]
        )

    with sub2:
        title_status = st.selectbox(
            "Trạng thái giấy tờ",
            categories["title_status"]
        )

    left, right = st.columns([1, 2], gap="large")

    with left:
        predict_btn = st.button("TÍNH TOÁN GIÁ TRỊ")

    with right:
        if predict_btn:

            if not location_input.strip():
                st.warning("⚠️ Vui lòng nhập vị trí niêm yết.")
                st.stop()

            if odometer <= 0:
                st.warning("⚠️ Số ODO phải lớn hơn 0.")
                st.stop()

            try:
                with st.spinner("Đang phân tích dữ liệu..."):

                    try:
                        location = geolocator.geocode(
                            location_input,
                            timeout=10
                        )
                    except (
                        GeocoderTimedOut,
                        GeocoderServiceError
                    ):
                        location = None

                    if location:
                        lat = location.latitude
                        long = location.longitude
                    else:
                        st.warning("""
⚠️ Không thể xác định vị trí chính xác.

Hệ thống sẽ sử dụng tọa độ mặc định để tiếp tục dự đoán 38.0 ,-95.0.
""")
                        lat, long = 38.0, -95.0

                    input_df = pd.DataFrame([{
                        "year": year,
                        "manufacturer": manufacturer,
                        "condition": condition,
                        "fuel": fuel,
                        "odometer": odometer,
                        "title_status": title_status,
                        "transmission": transmission,
                        "drive": drive,
                        "type": type_car,
                        "paint_color": paint_color,
                        "lat": lat,
                        "long": long
                    }])

                    price = np.expm1(
                        model.predict(input_df)[0]
                    )

                    st.success(f"💰 Giá dự đoán: ${price:,.0f} USD")

            except Exception as e:
                st.error(f"""
❌ Có lỗi xảy ra khi dự đoán.

Chi tiết lỗi:
{str(e)}
""")
                
def clean_numeric_data(val, col_name):
    if pd.isna(val) or val == "": return 0.0
    s = str(val).strip().replace(' ', '').replace('###', '')
    
    if s.count('.') > 1:
        last_dot_idx = s.rfind('.')
        before = s[:last_dot_idx].replace('.', '').replace(',', '')
        after = s[last_dot_idx:]
        s = before + after
    else:
        s = s.replace(',', '')

    try:
        num = float(s)
        if col_name in ['lat', 'long']:
            if abs(num) > 1000: return num / 10000
            if abs(num) > 180: return num / 10
        return num
    except:
        return 0.0


with tab_batch:
    st.markdown("""
    <div class="note-box">
        <h3 style='margin-top:0; color: #38bdf8;'>📋 HƯỚNG DẪN CHUẨN BỊ FILE DỮ LIỆU</h3>
        <p>Để hệ thống dự đoán chính xác nhất, vui lòng kiểm tra file CSV của bạn theo các quy tắc sau:</p>
    </div>
    """, unsafe_allow_html=True)

    guide_c1, guide_c2 = st.columns(2)

    with guide_c1:
        st.markdown("""
        **1. Các cột bắt buộc (12 cột):**
        - year, manufacturer, condition, fuel
        - odometer, title_status, transmission, drive
        - type, paint_color, lat, long
        
        **2. Định dạng chữ (Categorical):**
        - Phải trùng khớp với các lựa chọn trong phần **Dự đoán đơn lẻ**.
        - Ví dụ: manufacturer nên là toyota, honda, ford...
        """)

    with guide_c2:
        st.markdown("""
        **3. Định dạng số (Quan trọng):**
        - **Odometer:** Nhập số nguyên liền mạch (VD: 50000).
        - **Dữ liệu lỗi:** Nếu dòng nào chứa chữ ở cột số (VD: 'không biết'), hệ thống sẽ đưa về 0 để tránh lỗi ứng dụng.
        """)
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Tải lên file CSV dữ liệu xe",
        type=["csv"]
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("🔍 Xem trước dữ liệu vừa tải lên:")
            check_df = pd.DataFrame({
                "Cột": df.columns,
                "Kiểu dữ liệu hiện tại": df.dtypes.astype(str),
                "Trạng thái": ["✅" if col in df.columns else "❌ Thiếu" for col in df.columns]
            }).head(12) 
            st.table(check_df)

            if st.button("CHẠY DỰ ĐOÁN HÀNG LOẠT"):
                required_cols = [
                    "year", "manufacturer", "condition", "fuel", "odometer",
                    "title_status", "transmission", "drive", "type",
                    "paint_color", "lat", "long"
                ]

                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"❌ File thiếu các cột: {', '.join(missing_cols)}")
                else:
                    with st.spinner("Đang làm sạch dữ liệu và dự đoán..."):
                        X_process = df[required_cols].copy()

                        numeric_cols = ["year", "odometer", "lat", "long"]
                        for col in numeric_cols:
                            X_process[col] = X_process[col].apply(lambda x: clean_numeric_data(x, col))

                        try:
                            preds = np.expm1(model.predict(X_process))
                            
                            df["Predicted Price"] = np.round(preds, 0).astype(int)
                            
                            df.loc[df["Predicted Price"] > 1000000, "Predicted Price"] = 50000
                            df.loc[df["Predicted Price"] < 0, "Predicted Price"] = 0

                            st.success("✅ Xử lý thành công! Giá đã được làm tròn.")
                            st.dataframe(df, use_container_width=True)

                            st.download_button(
                                "📥 Tải kết quả (.csv)",
                                df.to_csv(index=False).encode("utf-8"),
                                "car_predictions_results.csv",
                                "text/csv"
                            )
                        except Exception as e:
                            st.error(f"❌ Lỗi trong quá trình tính toán mô hình: {str(e)}")

        except Exception as e:
            st.error(f"❌ Lỗi định dạng file CSV: {str(e)}")

st.markdown("""
<div class="footer">
    Car Price Prediction System
</div>
""", unsafe_allow_html=True) 