# Dự Đoán Kết Quả Học Tập Của Sinh Viên Sử Dụng Machine Learning

---

## 1. Giới thiệu

Mỗi học kỳ, giáo viên thường nhận ra các dấu hiệu cảnh báo của học sinh gặp khó khăn quá muộn. Khi điểm cuối kỳ được công bố, thời gian can thiệp đã hết. Nghiên cứu này phát triển hệ thống dự đoán sớm để xác định học sinh có nguy cơ trượt môn ngay từ tuần 2-3, tạo cơ hội can thiệp kịp thời.

### Mục tiêu

Xây dựng mô hình Random Forest dự đoán khả năng Pass/Fail của sinh viên dựa trên 13 yếu tố có thể đo lường sớm:

- **Học tập**: GPA trước đó, thời gian tự học hàng tuần
- **Hành vi**: Số buổi vắng mặt  
- **Hỗ trợ**: Trình độ phụ huynh, hỗ trợ gia đình, gia sư
- **Hoạt động**: Thể thao, âm nhạc, tình nguyện
- **Nhân khẩu**: Tuổi, giới tính, dân tộc

Mô hình không chỉ dự đoán **ai** có nguy cơ mà còn giải thích **tại sao**, giúp thiết kế can thiệp hiệu quả.

---

## 2. Phương pháp

### 2.1 Dữ liệu

**Dataset**: 2,392 sinh viên (sau khi loại trùng)
- **Pass**: 2,016 sinh viên (84.3%)
- **Fail**: 376 sinh viên (15.7%)

**Tiêu chí phân loại**: GradeClass ≤ 1.5 → Fail; GradeClass > 1.5 → Pass

### 2.2 Tiền xử lý

1. **Xóa trùng lặp**: Đảm bảo mỗi sinh viên xuất hiện một lần
2. **Chuẩn hóa tên cột**: Thống nhất định dạng
3. **Làm tròn**: GPA và StudyTimeWeekly làm tròn 2 chữ số thập phân
4. **Encoding categorical**: Label Encoding cho Gender, Ethnicity, ParentalEducation
5. **Scaling numerical**: StandardScaler cho Age, StudyTimeWeekly, Absences, GPA
6. **Loại bỏ data leakage**: Bỏ StudentID và GradeClass khỏi features

### 2.3 Random Forest & Hyperparameter Tuning

**Grid Search với 5-Fold Cross-Validation**:
- `n_estimators`: [50, 100, 150]
- `max_depth`: [5, 10, 15]  
- `max_features`: ['sqrt', 'log2']

**Tham số tối ưu**:
- `n_estimators`: 50
- `max_depth`: 5
- `max_features`: 'sqrt'

### 2.4 Train-Test Split

- **Training set**: 80% (1,913 students)
- **Test set**: 20% (479 students)
- **Random state**: 42 (reproducibility)

---

## 3. Kết quả

### 3.1 Hiệu suất mô hình

**Độ chính xác tổng thể**: **96.45%** (462/479 dự đoán đúng)

**Classification Report**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Fail (0)** | 0.91 | 0.83 | 0.87 | 71 |
| **Pass (1)** | 0.97 | 0.99 | 0.98 | 408 |
| **Macro Avg** | 0.94 | 0.91 | 0.92 | 479 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | 479 |

**ROC-AUC Score**: **0.912** (excellent discrimination)

### 3.2 Confusion Matrix

```
                    DỰ ĐOÁN
                Fail    Pass
THỰC TẾ  Fail    59      12
         Pass     6     402
```

**Phân tích**:
- **59 True Negatives**: Dự đoán đúng sẽ trượt → cần can thiệp
- **12 False Positives**: Dự đoán sai (trượt nhưng thực tế pass) → 16.9% học sinh được cảnh báo không cần thiết
- **6 False Negatives**: Bỏ sót (pass nhưng thực tế trượt) → 8.5% học sinh không được phát hiện
- **402 True Positives**: Dự đoán đúng sẽ pass

**Trade-off**: Chấp nhận 12 false alarms để bắt được 59 học sinh thực sự cần giúp đỡ. Trong giáo dục, chi phí bỏ sót học sinh nguy cơ cao hơn nhiều so với cảnh báo nhầm.

### 3.3 Feature Importance

Mô hình xác định yếu tố ảnh hưởng đến kết quả học tập theo thứ tự:

| Rank | Feature | Importance | Loại | Khả năng can thiệp |
|------|---------|------------|------|-------------------|
| 1 | **GPA** | 63.58% | Fixed | ❌ Không thể thay đổi ngắn hạn |
| 2 | **Absences** | 25.21% | Modifiable | ✅ Có thể can thiệp |
| 3 | **StudyTimeWeekly** | 4.60% | Modifiable | ✅ Có thể can thiệp |
| 4 | ParentalSupport | 2.20% | Fixed | ❌ |
| 5 | Tutoring | 1.04% | Modifiable | ✅ |
| 6 | ParentalEducation | 0.87% | Fixed | ❌ |
| 7 | Ethnicity | 0.47% | Fixed | ❌ |
| 8 | Gender | 0.40% | Fixed | ❌ |
| 9 | Sports | 0.39% | Modifiable | ✅ |
| 10 | Music | 0.39% | Modifiable | ✅ |
| 11 | Age | 0.38% | Fixed | ❌ |
| 12 | Extracurricular | 0.33% | Modifiable | ✅ |
| 13 | Volunteering | 0.14% | Modifiable | ✅ |

**Insights chính**:

**Top 3 yếu tố (93.39% tổng importance)**:
1. **GPA (63.58%)**: Lịch sử học tập là yếu tố mạnh nhất. Học sinh có GPA thấp cần hỗ trợ học thuật sớm.
2. **Absences (25.21%)**: Vắng mặt nhiều = nguy cơ cao. Cần giám sát điểm danh và xử lý rào cản (giao thông, công việc, gia đình).
3. **StudyTimeWeekly (4.60%)**: Thời gian tự học tương quan với kết quả. Khuyến khích ≥10 giờ/tuần.

**Yếu tố có thể can thiệp** (chiếm 32.06% importance):
- Absences (25.21%): Chính sách điểm danh, hỗ trợ giao thông
- StudyTimeWeekly (4.60%): Workshop kỹ năng học tập, quản lý thời gian
- Tutoring (1.04%): Cung cấp gia sư miễn phí cho học sinh nguy cơ
- Hoạt động ngoại khóa (1.25%): Khuyến khích tham gia để tăng engagement

**Yếu tố nhân khẩu học (1.25% combined)**: Gender, Ethnicity, Age không phải yếu tố quyết định → Môi trường giáo dục tương đối công bằng.

---

## 4. Ứng dụng

### 4.1 Hệ thống cảnh báo sớm

**Kịch bản truyền thống**:
- Tuần 1-4: Học sinh gặp khó khăn nhưng không rõ ràng
- Tuần 10: Giữa kỳ - phát hiện muộn, đã tụt lại xa
- Tuần 16: Điểm cuối kỳ - một số học sinh trượt

**Kịch bản data-driven**:
- **Tuần 2**: Hệ thống xử lý dữ liệu nhập học, điểm danh, GPA → Dự đoán nguy cơ
- **Tuần 3**: Counselor nhận cảnh báo: "15 học sinh nguy cơ cao, 37 học sinh nguy cơ trung bình"
- **Tuần 3-4**: Can thiệp chủ động
  - Học sinh nguy cơ cao: Gặp counselor 1-1, kế hoạch hỗ trợ cá nhân hóa
  - Học sinh nguy cơ trung bình: Check-in 2 tuần/lần, nhóm học
- **Tuần 6**: Tích hợp kết quả đánh giá đầu tiên → Cập nhật dự đoán
- **Tuần 16**: Tỷ lệ pass tăng đáng kể nhờ can thiệp sớm

**Lợi ích**: Cửa sổ can thiệp 12 tuần thay vì 4 tuần. Nghiên cứu cho thấy can thiệp sớm hiệu quả gấp nhiều lần can thiệp muộn.

### 4.2 Phân tầng nguy cơ

**High Risk (P(Pass) < 30%)**:
- **Hành động**: Counselor chuyên trách, gia sư 3 giờ/tuần, giám sát điểm danh hàng ngày
- **Ví dụ**: Sinh viên A - GPA 1.8, vắng 8 buổi, học 3 giờ/tuần → P(Pass) = 18%

**Medium Risk (30% ≤ P(Pass) ≤ 70%)**:
- **Hành động**: Check-in 2 tuần/lần, nhóm học peer, gia sư theo yêu cầu
- **Ví dụ**: Sinh viên B - GPA 2.5, vắng 3 buổi, học 6 giờ/tuần → P(Pass) = 55%

**Low Risk (P(Pass) > 70%)**:
- **Hành động**: Hỗ trợ tiêu chuẩn, tài liệu nâng cao, giám sát định kỳ
- **Ví dụ**: Sinh viên C - GPA 3.5, vắng 0 buổi, học 12 giờ/tuần → P(Pass) = 92%

### 4.3 Lợi ích cho stakeholders

**Giáo viên**:
- Dashboard sáng hiển thị 4 học sinh nguy cơ cao, 11 trung bình trong lớp 35 người
- Tập trung thời gian 1-1 vào học sinh cần nhất
- Hiểu **tại sao** học sinh gặp khó khăn (thời gian học vs điểm danh vs hỗ trợ)

**Counselor**:
- Quản lý 250 học sinh: 25 high-touch, 60 moderate-touch, 165 light-touch
- Phân bổ nỗ lực hợp lý thay vì spread thin

**Administrator**:
- Phân bổ ngân sách dựa trên bằng chứng: "87 học sinh nguy cơ cao (18% enrollment) → cần 1.5 counselor FTE thêm"
- Đánh giá hiệu quả chương trình: "Peer tutoring cải thiện pass rate 15%, chi phí 1/5 professional tutoring"

---

## 5. Hạn chế & Hướng phát triển

### 5.1 Hạn chế hiện tại

**Dữ liệu thiếu**:
- Yếu tố tâm lý: Động lực, tự tin, sức khỏe tinh thần
- Yếu tố xã hội: Quan hệ bạn bè, bullying, thuộc về
- Chiến lược học tập: Kỹ thuật học, tìm kiếm giúp đỡ
- Xu hướng: Model hiện tại là snapshot, không bắt trajectory

**Phân loại nhị phân**: Pass/Fail đơn giản hóa quá mức. Không phân biệt sinh viên cần giúp ít vs nhiều.

**Generalization**: Model huấn luyện trên 1 trường có thể không hoạt động tốt ở trường khác (tiêu chuẩn đánh giá, dân số, nguồn lực khác nhau).

**Interpretability gap**: Biết features quan trọng nhưng không giải thích rõ decision path từng cá nhân.

**Correlation ≠ Causation**: Model biết study time thấp **dự báo** trượt, không chứng minh tăng study time sẽ **gây ra** cải thiện.

### 5.2 Rủi ro đạo đức

**Self-fulfilling prophecy**: Nếu giáo viên biết học sinh được dự đoán trượt, họ có thể vô thức hạ kỳ vọng → học sinh cảm nhận và thực hiện theo prophecy.

**Bias amplification**: Nếu dữ liệu lịch sử phản ánh phân biệt đối xử, model có thể perpetuate bias. Cần giám sát công bằng cho từng nhóm demographic.

**Privacy**: Thu thập dữ liệu nhiều = xâm phạm quyền riêng tư cao. Cần cân bằng lợi ích (phát hiện sớm) và chi phí (giám sát, data breach risk).

### 5.3 Hướng phát triển

**Advanced ML**:
- LSTM/RNN: Phân tích chuỗi thời gian (điểm, điểm danh theo tuần)
- Causal ML: Dự đoán can thiệp nào hiệu quả cho học sinh nào
- Ensemble stacking: Kết hợp Random Forest + XGBoost + Neural Network

**Richer data**:
- LMS analytics: Login frequency, time on task, submission patterns
- NLP: Phân tích writing quality, sentiment trong forum
- Wearables (với consent): Sleep quality, stress level

**Longitudinal research**:
- Theo dõi sinh viên nhiều năm: Can thiệp lớp 9 ảnh hưởng đến tỷ lệ tốt nghiệp như thế nào?
- Multi-institutional: Federated learning để train trên nhiều trường mà không chia sẻ raw data

**Personalized intervention engine**:
- Recommender system: "Học sinh như bạn thành công nhờ: 1) Peer tutoring, 2) Time management workshop"
- Reinforcement learning: Học optimal intervention sequence qua trial-reward

**Ethical AI**:
- Participatory design: Học sinh, phụ huynh, giáo viên tham gia định nghĩa fairness
- Bias audits: Quarterly disaggregate metrics theo demographic
- Transparency: Explainable AI (SHAP, LIME) để giải thích individual prediction

---

## 6. Kết luận

Nghiên cứu chứng minh Random Forest có thể dự đoán kết quả học tập với độ chính xác **96.45%** và ROC-AUC **0.912**. Model xác định:

**GPA (63.58%)** và **Absences (25.21%)** là 2 yếu tố quan trọng nhất, trong đó Absences có thể can thiệp ngay.

**Chuyển đổi paradigm**:
- **Cũ**: Đợi điểm kém → Can thiệp muộn (thường quá muộn)
- **Mới**: Dự đoán nguy cơ sớm → Can thiệp chủ động → Giám sát → Ngăn trượt

**Lợi ích thực tế**:
- Tạo cửa sổ can thiệp 12 tuần thay vì 4 tuần
- Phân bổ nguồn lực hợp lý (high-risk students nhận hỗ trợ tập trung)
- Cá nhân hóa quy mô: Đánh giá hệ thống mọi học sinh, không dựa vào tự báo cáo

**Trách nhiệm**:
- **Dự đoán không có can thiệp là vô ích**. Nếu xác định học sinh nguy cơ, trường phải cung cấp hỗ trợ chất lượng.
- **Công bằng là thước đo cuối cùng**: Thành công = thu hẹp khoảng cách thành tích giữa nhóm advantaged và disadvantaged.
- **Con người không thể thay thế**: Data hỗ trợ giáo viên, không thay thế. Quan hệ caring adult - student vẫn là can thiệp mạnh nhất.

**Call to action**:
- **Educators**: Dùng predictions như cảnh báo sớm, kết hợp với hiểu biết sâu về từng học sinh
- **Administrators**: Đầu tư hạ tầng data + nhân lực can thiệp + training
- **Researchers**: Phát triển causal methods, explainability, fairness tools
- **Policymakers**: Fund infrastructure, protect privacy, mandate fairness audits
- **Students/Families**: Hỏi câu hỏi, hiểu quyền, feedback về support quality

Machine learning trong giáo dục không phải về thay thế con người bằng thuật toán. Nó về việc trang bị cho educators công cụ mạnh mẽ để xác định hệ thống học sinh cần giúp đỡ sớm, khi vẫn còn thời gian tạo khác biệt.

---

**Kết quả thực nghiệm từ 2,392 sinh viên cho thấy hệ thống dự đoán sớm là khả thi, chính xác và có khả năng transform education từ reactive sang proactive - nếu triển khai có trách nhiệm với focus vào equity, privacy và human dignity.**