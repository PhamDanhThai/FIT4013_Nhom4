"""FIT4013 - Đề tài 15: Dự đoán điểm chất lượng rượu vang bằng hồi quy.

Chạy từ thư mục gốc repository:
    python FIT4013_DeTai15/src/train.py

Dữ liệu: Wine Quality, UCI Machine Learning Repository.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

THU_MUC_GOC = Path(__file__).resolve().parents[2]
URL_DU_LIEU = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
CAC_TEP_DU_LIEU = [("Đỏ", "winequality-red.csv", 1), ("Trắng", "winequality-white.csv", 0)]
TEN_MO_HINH = {"Baseline": "Mô hình cơ sở", "KNN": "KNN", "Tree": "Cây quyết định"}
TEN_THUOC_TINH = {
    "fixed acidity": "Độ axit cố định",
    "volatile acidity": "Độ axit bay hơi",
    "citric acid": "Axit citric",
    "residual sugar": "Đường dư",
    "chlorides": "Hàm lượng clorua",
    "free sulfur dioxide": "SO₂ tự do",
    "total sulfur dioxide": "Tổng SO₂",
    "density": "Khối lượng riêng",
    "pH": "Độ pH",
    "sulphates": "Hàm lượng sunfat",
    "alcohol": "Nồng độ cồn",
    "quality": "Điểm chất lượng",
    "wine_type": "Loại rượu (đỏ = 1, trắng = 0)",
}
TEN_THONG_KE = {"count": "Số lượng", "mean": "Trung bình", "std": "Độ lệch chuẩn",
                 "min": "Nhỏ nhất", "25%": "Phân vị 25%", "50%": "Trung vị",
                 "75%": "Phân vị 75%", "max": "Lớn nhất"}


def cau_hinh_bieu_do() -> None:
    """Đặt phông hỗ trợ tiếng Việt có dấu trên hầu hết hệ điều hành."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.autolayout": True,
    })


def tai_du_lieu(thu_muc: Path) -> pd.DataFrame:
    """Đọc dữ liệu có sẵn; chỉ tải từ UCI nếu thiếu tệp."""
    thu_muc.mkdir(parents=True, exist_ok=True)
    cac_bang = []
    for ten_loai, ten_tep, ma_loai in CAC_TEP_DU_LIEU:
        duong_dan = thu_muc / ten_tep
        if not duong_dan.exists():
            print(f"Đang tải dữ liệu rượu vang {ten_loai.lower()}: {ten_tep}")
            try:
                urllib.request.urlretrieve(URL_DU_LIEU + ten_tep, duong_dan)
            except Exception as loi:
                raise RuntimeError(
                    f"Không thể tải {ten_tep}. Hãy kiểm tra Internet hoặc chép tệp vào {thu_muc}."
                ) from loi
        bang = pd.read_csv(duong_dan, sep=";")
        bang["wine_type"] = ma_loai
        cac_bang.append(bang)
    ket_qua = pd.concat(cac_bang, ignore_index=True)
    if "quality" not in ket_qua.columns:
        raise ValueError("Bộ dữ liệu phải có cột 'quality' (điểm chất lượng).")
    print(f"Đã đọc {len(ket_qua):,} mẫu; {ket_qua.shape[1] - 1} đặc trưng đầu vào.")
    return ket_qua


def tao_pipeline(mo_hinh, phuong_an: str) -> Pipeline:
    """A: chuẩn hóa StandardScaler; B: RobustScaler và chọn 8 đặc trưng."""
    cac_buoc = [("imputer", SimpleImputer(strategy="median"))]
    if phuong_an == "A":
        cac_buoc.append(("scale", StandardScaler()))
    elif phuong_an == "B":
        cac_buoc.extend([
            ("scale", RobustScaler()),
            ("select", SelectKBest(score_func=f_regression, k=8)),
        ])
    else:
        raise ValueError("Phương án xử lý phải là A hoặc B.")
    cac_buoc.append(("model", mo_hinh))
    return Pipeline(cac_buoc)


def luu_csv(bang: pd.DataFrame, duong_dan: Path, co_chi_muc: bool = False) -> None:
    """UTF-8 BOM giúp Excel/WPS trên Windows hiển thị tiếng Việt chính xác."""
    bang.to_csv(duong_dan, index=co_chi_muc, encoding="utf-8-sig")


def thong_ke_du_lieu(du_lieu: pd.DataFrame, dau_ra: Path) -> None:
    """Xuất kết quả gốc để tương thích và bản dịch cho người đọc."""
    thong_ke = du_lieu.describe(include="all")
    thong_ke.to_csv(dau_ra / "descriptive_statistics.csv")
    ban_dich = thong_ke.rename(index=TEN_THONG_KE, columns=TEN_THUOC_TINH)
    luu_csv(ban_dich, dau_ra / "thong_ke_mo_ta_vi.csv", co_chi_muc=True)

    thieu = du_lieu.isna().sum().rename("so_gia_tri_thieu")
    thieu.to_csv(dau_ra / "missing_values.csv")
    luu_csv(pd.DataFrame({"Thuộc tính": [TEN_THUOC_TINH.get(c, c) for c in thieu.index],
                          "Số giá trị thiếu": thieu.to_numpy()}), dau_ra / "gia_tri_thieu_vi.csv")

    phan_phoi = du_lieu["quality"].value_counts().sort_index()
    phan_phoi.to_csv(dau_ra / "quality_distribution.csv")
    luu_csv(pd.DataFrame({"Điểm chất lượng": phan_phoi.index,
                          "Số lượng mẫu": phan_phoi.to_numpy()}), dau_ra / "phan_phoi_chat_luong_vi.csv")

    with (dau_ra / "data_info.txt").open("w", encoding="utf-8") as tep:
        tep.write("THÔNG TIN BỘ DỮ LIỆU CHẤT LƯỢNG RƯỢU VANG\n")
        tep.write(f"Số mẫu: {len(du_lieu)}\nSố thuộc tính: {du_lieu.shape[1]}\n")
        tep.write("\nTên thuộc tính và kiểu dữ liệu:\n")
        for cot, kieu in du_lieu.dtypes.items():
            tep.write(f"- {TEN_THUOC_TINH.get(cot, cot)} ({cot}): {kieu}\n")
        tep.write(f"\nTổng giá trị thiếu: {int(du_lieu.isna().sum().sum())}\n")

    fig, truc = plt.subplots(figsize=(9, 5))
    truc.bar(phan_phoi.index.astype(str), phan_phoi.values)
    truc.set(xlabel="Điểm chất lượng", ylabel="Số lượng mẫu",
             title="Phân phối điểm chất lượng rượu vang")
    fig.savefig(dau_ra / "quality_distribution.png", dpi=170)
    plt.close(fig)
    print("Đã lưu thống kê mô tả, giá trị thiếu và phân phối chất lượng.")


def cau_hinh_mo_hinh():
    tham_so_knn = {"model__n_neighbors": [3, 5, 9, 15],
                   "model__weights": ["uniform", "distance"]}
    tham_so_cay = {"model__max_depth": [4, 8, 12, None],
                   "model__min_samples_leaf": [1, 5, 10]}
    return [
        ("Baseline", "A", DummyRegressor(strategy="mean"), {}),
        ("KNN", "A", KNeighborsRegressor(), tham_so_knn),
        ("KNN", "B", KNeighborsRegressor(), tham_so_knn),
        ("Tree", "A", DecisionTreeRegressor(random_state=42), tham_so_cay),
        ("Tree", "B", DecisionTreeRegressor(random_state=42), tham_so_cay),
    ]


def viet_hoa_bang_chi_so(bang: pd.DataFrame) -> pd.DataFrame:
    ban_dich = bang.rename(columns={
        "model": "Mô hình", "pipeline": "Phương án", "cv_mae": "MAE kiểm định chéo",
        "test_mae": "MAE tập kiểm tra", "test_rmse": "RMSE tập kiểm tra",
        "test_r2": "R² tập kiểm tra", "best_params": "Siêu tham số tốt nhất",
    }).copy()
    ban_dich["Mô hình"] = ban_dich["Mô hình"].map(TEN_MO_HINH)
    return ban_dich


def ve_bieu_do_danh_gia(bang: pd.DataFrame, thuc_te: pd.Series,
                         du_doan: np.ndarray, ten_tot_nhat: str, dau_ra: Path) -> None:
    nhan = [f"{TEN_MO_HINH[m]} - {p}" for m, p in zip(bang["model"], bang["pipeline"])]
    fig, truc = plt.subplots(figsize=(10, 5))
    truc.bar(nhan, bang["test_mae"])
    truc.set(ylabel="Sai số tuyệt đối trung bình (MAE)",
             title="So sánh MAE trên tập kiểm tra giữa các mô hình")
    truc.tick_params(axis="x", rotation=15)
    fig.savefig(dau_ra / "mae_comparison.png", dpi=170)
    plt.close(fig)

    sai_so = np.asarray(thuc_te) - du_doan
    fig, truc = plt.subplots(figsize=(9, 5))
    truc.hist(sai_so, bins=25, edgecolor="black")
    truc.set(xlabel="Sai số (thực tế − dự đoán)", ylabel="Tần suất",
             title=f"Phân phối sai số: {ten_tot_nhat}")
    fig.savefig(dau_ra / "error_distribution.png", dpi=170)
    plt.close(fig)

    fig, truc = plt.subplots(figsize=(7, 6))
    truc.scatter(thuc_te, du_doan, alpha=0.25)
    gioi_han = [min(thuc_te.min(), du_doan.min()), max(thuc_te.max(), du_doan.max())]
    truc.plot(gioi_han, gioi_han, "k--", label="Dự đoán hoàn hảo")
    truc.set(xlabel="Điểm chất lượng thực tế", ylabel="Điểm chất lượng dự đoán",
             title=f"Giá trị thực tế so với dự đoán: {ten_tot_nhat}")
    truc.legend()
    fig.savefig(dau_ra / "actual_vs_predicted.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán điểm chất lượng rượu vang - Đề tài 15, FIT4013")
    parser.add_argument("--data-dir", type=Path, default=THU_MUC_GOC / "data",
                        help="Thư mục chứa dữ liệu rượu vang (mặc định: data/ của repository)")
    parser.add_argument("--out-dir", type=Path, default=THU_MUC_GOC / "results",
                        help="Thư mục chứa các kết quả (mặc định: results/ của repository)")
    args = parser.parse_args()
    dau_ra = args.out_dir
    dau_ra.mkdir(parents=True, exist_ok=True)
    cau_hinh_bieu_do()
    print("=== NHẬP MÔN HỌC MÁY - ĐỀ TÀI 15: DỰ ĐOÁN CHẤT LƯỢNG RƯỢU VANG ===")
    du_lieu = tai_du_lieu(args.data_dir)
    thong_ke_du_lieu(du_lieu, dau_ra)

    x = du_lieu.drop(columns="quality")
    y = du_lieu["quality"]
    x_huan_luyen, x_kiem_tra, y_huan_luyen, y_kiem_tra = train_test_split(
        x, y, test_size=0.2, random_state=42)
    print(f"Chia dữ liệu: {len(x_huan_luyen)} mẫu huấn luyện, {len(x_kiem_tra)} mẫu kiểm tra.")
    chia_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cac_ket_qua = []
    cac_du_doan = {}

    for ten, phuong_an, mo_hinh, tham_so in cau_hinh_mo_hinh():
        khoa = f"{ten}_{phuong_an}"
        print(f"\nĐang huấn luyện và kiểm định chéo: {TEN_MO_HINH[ten]} - phương án {phuong_an}")
        tim_kiem = GridSearchCV(
            tao_pipeline(mo_hinh, phuong_an), tham_so, cv=chia_cv,
            scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
        tim_kiem.fit(x_huan_luyen, y_huan_luyen)
        du_doan = tim_kiem.predict(x_kiem_tra)
        cac_du_doan[khoa] = du_doan
        cac_ket_qua.append({
            "model": ten, "pipeline": phuong_an,
            "cv_mae": -float(tim_kiem.best_score_),
            "test_mae": mean_absolute_error(y_kiem_tra, du_doan),
            "test_rmse": float(np.sqrt(mean_squared_error(y_kiem_tra, du_doan))),
            "test_r2": r2_score(y_kiem_tra, du_doan),
            "best_params": json.dumps(tim_kiem.best_params_, ensure_ascii=False),
        })

    # Chọn mô hình theo MAE kiểm định chéo, KHÔNG dựa vào kết quả tập kiểm tra.
    bang = pd.DataFrame(cac_ket_qua).sort_values("cv_mae").reset_index(drop=True)
    luu_csv(bang, dau_ra / "metrics.csv")
    bang_tieng_viet = viet_hoa_bang_chi_so(bang)
    luu_csv(bang_tieng_viet, dau_ra / "chi_so_danh_gia_vi.csv")

    hang_tot_nhat = bang.iloc[0]
    khoa_tot_nhat = f"{hang_tot_nhat['model']}_{hang_tot_nhat['pipeline']}"
    ten_tot_nhat = f"{TEN_MO_HINH[hang_tot_nhat['model']]} - {hang_tot_nhat['pipeline']}"
    du_doan_tot_nhat = cac_du_doan[khoa_tot_nhat]
    sai_so = np.asarray(y_kiem_tra) - du_doan_tot_nhat
    bang_sai_so = pd.DataFrame({
        "actual": np.asarray(y_kiem_tra), "predicted": du_doan_tot_nhat,
        "error": sai_so, "abs_error": np.abs(sai_so),
    }).sort_values("abs_error", ascending=False)
    luu_csv(bang_sai_so, dau_ra / "prediction_errors.csv")
    luu_csv(bang_sai_so.rename(columns={
        "actual": "Giá trị thực tế", "predicted": "Giá trị dự đoán",
        "error": "Sai số", "abs_error": "Sai số tuyệt đối",
    }), dau_ra / "sai_so_du_doan_vi.csv")
    ve_bieu_do_danh_gia(bang, y_kiem_tra, du_doan_tot_nhat, ten_tot_nhat, dau_ra)

    thong_tin_chay = {
        "nguon_du_lieu": "UCI Wine Quality", "so_mau": len(du_lieu),
        "so_dac_trung": x.shape[1], "so_mau_huan_luyen": len(x_huan_luyen),
        "so_mau_kiem_tra": len(x_kiem_tra), "hat_ngau_nhien": 42,
        "bien_muc_tieu": "quality", "mo_hinh_tot_nhat_theo_cv_mae": khoa_tot_nhat,
        "ghi_chu": "Mô hình được chọn theo MAE kiểm định chéo trên tập huấn luyện; tập kiểm tra chỉ dùng để đánh giá.",
    }
    (dau_ra / "run_metadata.json").write_text(
        json.dumps(thong_tin_chay, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== BẢNG KẾT QUẢ ĐÁNH GIÁ ===")
    print(bang_tieng_viet.to_string(index=False))
    print(f"\nMô hình tốt nhất theo kiểm định chéo: {ten_tot_nhat}")
    print(f"MAE kiểm định chéo: {hang_tot_nhat['cv_mae']:.4f}")
    print(f"MAE tập kiểm tra: {hang_tot_nhat['test_mae']:.4f}")
    print(f"RMSE tập kiểm tra: {hang_tot_nhat['test_rmse']:.4f}")
    print(f"R² tập kiểm tra: {hang_tot_nhat['test_r2']:.4f}")
    print(f"\nĐã lưu toàn bộ kết quả tại: {dau_ra.resolve()}")


if __name__ == "__main__":
    main()
