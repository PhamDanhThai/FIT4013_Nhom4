# FIT4013 – Đề tài 15: Dự đoán điểm chất lượng rượu vang

**Nhóm 4 – CNTT 18-09**  
**Thành viên:** Phạm Danh Thái, Ngô Văn Hiếu, Phạm Anh Quân

## 1. Mục tiêu

Sử dụng các chỉ tiêu hóa lý của rượu vang để dự đoán điểm `quality` bằng hồi quy. Nhóm so sánh mô hình cơ sở, KNN và cây quyết định dưới hai phương án tiền xử lý; chọn mô hình bằng MAE kiểm định chéo và đánh giá trên tập kiểm tra.

## 2. Dữ liệu

Nguồn: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), Cortez và cộng sự (2009). Hai tệp đỏ/trắng được ghép và thêm biến `wine_type` (đỏ=1, trắng=0). Tên cột UCI giữ nguyên. Khi chạy lần đầu, Notebook tự tải hai tệp dữ liệu từ UCI nếu chưa có trong thư mục `data/` (cần Internet). Có thể tải thủ công từ nguồn UCI hoặc sao chép hai tệp dữ liệu thật từ repository GitHub của nhóm vào `data/`.

## 3. Phương pháp

- Chia train/test 80/20, `random_state=42`.
- Pipeline A: `SimpleImputer(median)` → `StandardScaler` → mô hình.
- Pipeline B: `SimpleImputer(median)` → `RobustScaler` → `SelectKBest(k=8)` → mô hình.
- Mô hình: `DummyRegressor`, `KNeighborsRegressor`, `DecisionTreeRegressor`.
- Tối ưu siêu tham số bằng `GridSearchCV` 5-fold trên train, chọn theo CV MAE; báo cáo MAE, RMSE, R² trên test.

## 4. Cài đặt và chạy

Yêu cầu Python 3.10+ (khuyến nghị 3.12). Mở terminal tại thư mục dự án:

```bash
python -m pip install -r requirements.txt
jupyter notebook DeTai15_WineQuality.ipynb
```

Chọn **Kernel → Restart & Run All**. Có thể chạy `python src/train.py --data-dir data --out-dir results` để tham khảo phiên bản dòng lệnh; Notebook là bản trình bày chính và có kết luận tiếng Việt theo số liệu thực nghiệm.

**Kiểm thử đã thực hiện:** Notebook chạy hết các ô trên dữ liệu kiểm thử mô phỏng 200 mẫu để kiểm tra kỹ thuật. Không dùng số liệu kiểm thử này làm kết quả nghiên cứu. Cần chạy lại trên toàn bộ dữ liệu UCI thật trước khi nộp.

## 5. Đọc kết quả

- `results/bang_so_sanh_mo_hinh.csv`: bảng các mô hình và chỉ số đánh giá.
- `results/so_sanh_sai_so_MAE.png`: so sánh MAE trên tập test.
- `results/phan_phoi_chat_luong_ruou.png`: phân phối nhãn.
- `results/phan_phoi_sai_so.png`: phân phối sai số.
- `results/thuc_te_va_du_doan.png`: dự đoán so với thực tế.
- `results/sai_so_theo_muc_diem.csv`: sai số theo từng mức chất lượng.
- `results/ket_luan_thuc_nghiem.txt`: kết luận được sinh từ lần chạy thực tế.

## 6. Giới hạn

Kết quả phụ thuộc vào bộ dữ liệu UCI và cách chia train/test. Điểm chất lượng không cân đối giữa các mức; chưa có dữ liệu độc lập bên ngoài để đánh giá khả năng tổng quát. So sánh test của các mô hình chỉ mang tính báo cáo, không dùng để điều chỉnh chọn mô hình.

## 7. Tài liệu tham khảo

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
2. UCI Machine Learning Repository. *Wine Quality Dataset*. https://archive.ics.uci.edu/dataset/186/wine+quality
3. Scikit-learn User Guide. https://scikit-learn.org/stable/user_guide.html

**Lưu ý học thuật:** Nội dung thực nghiệm cần được nhóm kiểm tra và hiểu rõ trước khi nộp; không coi các chỉ số trên một lần chia dữ liệu là bảo đảm hiệu quả trong mọi tình huống.
