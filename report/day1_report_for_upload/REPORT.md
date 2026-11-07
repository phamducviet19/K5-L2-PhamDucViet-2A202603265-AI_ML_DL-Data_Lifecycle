# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 7/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.12.10 / 2.15.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh** (toàn bộ khung hình).
- **Lý do:** Tác vụ là phân loại ảnh (`image_classification`). Trong record chỉ có `rank`, `class_id`, `class_name` và `score`, hoàn toàn không có trường tọa độ vị trí (không có `bbox_xyxy` hay đa giác `polygon_xy`). Mô hình nhận toàn bộ ma trận điểm ảnh làm đầu vào và dự đoán phân phối xác suất trên toàn thể không gian ảnh, gán một nhãn phân loại cho cả bức ảnh chứ không định vị từng vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp **không phải do model tự nghĩ ra**, mà **đi kèm với taxonomy và checkpoint** được huấn luyện sẵn (cụ thể là taxonomy `ImageNet-1K` gồm 1000 lớp cố định đi kèm checkpoint `yolo11n-cls.pt`). Mô hình chỉ tính toán phân phối xác suất trên danh sách các lớp đã được định nghĩa trước này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần ghi đầy đủ cả ba trường để đảm bảo tính toàn vẹn và nguồn gốc dữ liệu (data provenance, reproducibility):
  - `class_id`: Là mã định danh số học cố định dùng trong xử lý máy tính, lập trình và pipeline huấn luyện, tránh lỗi chuỗi ký tự hay khác biệt cách viết hoa/thường/dấu cách.
  - `class_name`: Giúp con người (annotator, reviewer, kỹ sư) đọc hiểu trực quan ngữ nghĩa của nhãn.
  - `taxonomy_name`: Xác định chính xác không gian nhãn tham chiếu (ở đây là ImageNet-1K). Các bộ dữ liệu khác nhau (như COCO-80, Pascal VOC) có thể dùng cùng tên lớp (ví dụ "car", "chair") hoặc cùng một ID số nhưng phạm vi khái niệm (ontology) hoặc chỉ số lại khác nhau. Ghi rõ cả ba trường giúp hệ thống nhất quán, tránh nhầm lẫn khi tích hợp hay mở rộng dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ nguyên tắc chọn lớp ưu tiên:
  - **Tiêu chí xác định chủ thể chính (salient object):** Ưu tiên đối tượng chiếm diện tích lớn nhất, nằm ở vị trí trung tâm, hoặc là tâm điểm hành động trong bối cảnh ảnh.
  - **Thứ bậc ưu tiên (class priority / hierarchy):** Nếu có nhiều đối tượng cạnh tranh ngang nhau, quy định rõ thứ tự ưu tiên (ví dụ phương tiện giao thông công cộng > xe cá nhân > người đi bộ, hoặc cảnh quan chung).
  - **Quy trình xử lý ca mơ hồ:** Nếu không thể xác định được một chủ thể duy nhất, hướng dẫn người gán nhãn gắn cờ ảnh không đạt chuẩn (ambiguous) hoặc chuyển lên Lab Coach (escalation), không để người gán nhãn tự phỏng đoán.
6. Vì sao không được lấy `score` của model làm ground truth?
- Vì `score` (confidence score) chỉ là giá trị xác suất toán học/độ tự tin của mô hình dự đoán (prediction) dựa trên trọng số đã học, không phải là sự thật khách quan (ground truth). Mô hình hoàn toàn có thể tự tin sai (overconfident) với score cao hoặc nhận diện đúng nhưng score thấp do góc chụp/ánh sáng phức tạp. Ground truth bắt buộc phải là nhãn do con người thẩm định và gán nhãn chính xác theo bộ quy tắc (guideline) nghiêm ngặt.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Hệ tọa độ `bbox_xyxy` tính theo đơn vị pixel với gốc `(0, 0)` tại góc trên cùng bên trái của bức ảnh (kích thước ảnh 640 x 427):
  - **Góc trên-trái (`x_min`, `y_min`):** Nằm tại tọa độ pixel `(385.33, 69.24)` (cách mép trái ảnh 385.33 px và cách mép trên 69.24 px).
  - **Góc dưới-phải (`x_max`, `y_max`):** Nằm tại tọa độ pixel `(498.92, 348.92)` (cách mép trái 498.92 px và cách mép trên 348.92 px).
  - **Kích thước box:** Chiều rộng `bbox_width = 113.58` pixel (`498.92 - 385.33`) và chiều cao `bbox_height = 279.68` pixel (`348.92 - 69.24`). Box này bao quanh người đang đứng chuẩn bị đồ ăn tại bếp.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Theo kết quả chạy từ ô mã so sánh trong notebook:
- Ngưỡng **0.20**: Có **17** prediction (gồm: `person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`, `spoon`, `potted plant`, `spoon`, `dining table`, `spoon`, `bottle`).
- Ngưỡng **0.60**: Có **6** prediction (gồm: `person`, `bowl`, `bowl`, `oven`, `oven`, `person`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng (từ 0.60 xuống 0.20), số lượng prediction tăng vọt từ 6 lên 17 hộp. Người reviewer phải làm thêm các việc:
  - Mất nhiều thời gian và công sức kiểm tra, lọc bỏ các hộp dương tính giả (False Positives – phát hiện sai, box rác/nhiễu).
  - Rà soát và loại bỏ các hộp bị trùng lặp hoặc chồng chéo lên nhau trên cùng một vật thể do mô hình dự đoán nhiều box cùng lúc cho một đối tượng.
  - Căn chỉnh lại ranh giới các box có điểm số thấp vì chúng thường bị vẽ lỏng lẻo, cắt xén sai lệch hoặc gán nhầm lớp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- **Quy tắc Hộp bao khít phần nhìn thấy (Tight Visible Bounding Box):**
  > Bounding box phải ôm khít tối đa toàn bộ các pixel nhìn thấy được (visible boundary) của vật thể, không để khoảng trống thừa giữa mép box và mép vật thể quá 2-3 pixel, đồng thời không được cắt xén mất bất kỳ bộ phận nhìn thấy nào của vật thể. Tuyệt đối không tự ý phỏng đoán vẽ box bao phủ các phần bị che khuất hoàn toàn.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- **Chỉ ra vật thể:** Chiếc lò nướng / tủ bếp bên trái (`class_name = oven`, `bbox_xyxy = [0.11, 187.54, 195.96, 292.81]`) có mép trái `x = 0.11 px` chạm sát mép biên ảnh và bị các vật dụng phía trước che một phần; hoặc người thứ hai ở mép trái (`bbox = [0.12, 263.18, 61.55, 310.78]`) chỉ lộ một phần cánh tay/thân áo do bị cắt mép trái và bị quầy bếp che khuất; hoặc các chiếc bát (`bowl`) xếp chồng lên nhau ở quầy bếp.
- **Xử lý bằng Guideline hay Escalation? Vì sao?:**
  - **Đưa vào Guideline:** Vì hiện tượng vật thể bị cắt mép (truncated) và bị che khuất (occluded) là đặc tính phổ biến, lặp lại thường xuyên trong thị giác máy tính. Guideline tiêu chuẩn cần quy định rõ: *"Nếu vật thể bị cắt mép hoặc che khuất nhưng phần nhìn thấy vẫn nhận diện rõ ràng đặc trưng của lớp (> 20% diện tích) thì vẽ box ôm khít toàn bộ phần nhìn thấy chạm tới mép ảnh và đánh dấu thuộc tính occluded/truncated; nếu bị che khuất quá mức không thể nhận diện chắc chắn thì bỏ qua."*
  - Chỉ đưa lên **Escalation** đối với những trường hợp dị biệt không rõ ràng (ví dụ: người ở mép trái chỉ lộ một vệt mờ không thể xác định chắc chắn là người hay tạp dề/khăn lau), khi đó annotator dừng lại và xin quyết định từ Lab Coach để cập nhật quy tắc thống nhất vào guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Bounding box chỉ là một hình chữ nhật bao quanh vật thể xác định bằng 4 điểm cực trị (`x_min`, `y_min`, `x_max`, `y_max`), do đó chứa rất nhiều điểm ảnh nền (background) hoặc vật thể lân cận, không thể hiện được hình thái thực tế của đối tượng.
- Ngược lại, Polygon cung cấp chuỗi tọa độ khép kín định nghĩa chính xác đường biên mức pixel (pixel-level silhouette / contour):
  - Cho biết chính xác hình dáng thực tế (đường cong cơ thể, tư thế đứng/cúi, cánh tay, trang phục...).
  - Cung cấp diện tích thực và hình học chính xác của vật thể, tách biệt hoàn toàn đối tượng khỏi nền (background).
  - Phân định ranh giới tiếp xúc chính xác giữa các vật thể nằm cạnh hoặc đè lên nhau.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` (ví dụ `kitchen-001`, `kitchen-009`) dùng để phân biệt các cá thể (đối tượng riêng biệt) cùng xuất hiện trong một bức ảnh tĩnh.
- **Ý nghĩa:**
  - **Không phải class ID:** Class ID là mã chủng loại đối tượng (ví dụ class `person` có ID = 0, class `bowl` có ID = 45). Nếu chỉ dùng class ID thì mọi người trong ảnh đều mang mã 0, không phân biệt được người này với người kia.
  - **Không phải tracking ID:** Tracking ID là mã định danh xuyên suốt chuỗi thời gian (video tracking), dùng để theo dõi cùng một đối tượng di chuyển qua nhiều khung hình video liên tiếp. Trong khi đó, `instance_id` ở đây chỉ là mã định danh cục bộ trong phạm vi một ảnh tĩnh duy nhất (`sample_id`), được đánh số thứ tự phát hiện trong ảnh đó mà không có ý nghĩa liên kết thời gian qua các ảnh khác.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- **Quy tắc vẽ biên Polygon bám sát ranh giới pixel nhìn thấy (Precise Pixel-Boundary Alignment):**
  > Đường biên đa giác phải men theo đúng đường viền phân tách giữa vật thể và hậu cảnh/vật thể khác với sai số không quá 1-2 pixel; không được lẹm vào phần cơ thể/vật thể và không để lẫn pixel nền; tại các đường cong gấp khúc (như bàn tay, nếp gấp quần áo) phải đặt các điểm nút đa giác với mật độ dày hơn để phản ánh trung thực hình thái của đối tượng.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- **Chỉ ra vị trí:** Vùng các chiếc bát (`bowl`) đặt sát nhau và chồng đè lên nhau ở quầy bếp bên trái (`kitchen-002`, `kitchen-003`, `kitchen-006`); hoặc vị trí quầy bếp tiếp xúc và che khuất nửa thân dưới của người phụ nữ (`kitchen-001`); hoặc mép chiếc thìa (`spoon`) tiếp giáp với mặt bàn.
- **Dùng Guideline hay Escalation? Vì sao?:**
  - **Dùng Guideline:** Đối với các trường hợp vật thể chạm nhau hoặc che khuất một phần thông thường (như bát xếp cạnh nhau, người đứng sau quầy bếp), Guideline phải quy định nguyên tắc: *"Vật thể nằm ở lớp trên (foreground) được vẽ trọn vẹn đường biên; vật thể ở lớp dưới (background bị che) chỉ vẽ phần ranh giới nhìn thấy được (visible mask), tuyệt đối không vẽ dự đoán phần bị che khuất."*
  - **Dùng Escalation:** Chỉ áp dụng khi gặp vùng biên quá nhòe mờ do chuyển động (motion blur), phản xạ ánh sáng chói, hoặc vùng bóng tối mà mắt người không thể phân định được ranh giới thuộc về vật nào. Khi đó annotator phải dừng lại, chụp ảnh và báo cáo Lab Coach để nhận chỉ đạo thống nhất, tránh tự ý suy đoán làm hỏng chất lượng ground truth.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục duy nhất (`class_id`, `class_name` thuộc taxonomy ImageNet-1K) gán cho toàn bộ bức ảnh, không có tọa độ không gian. | Trên ảnh `traffic`, khung cảnh giao thông phức tạp có nhiều vật thể (xe buýt, ô tô con, người đi bộ...). Model dự đoán top-1 là `cab` với score thấp (0.5109), top-2 là `minibus` (0.1643). Rất khó quyết định 1 nhãn đại diện cho toàn bộ ảnh vì không có chủ thể độc tôn. | Đọc kỹ guideline để nắm quy tắc ưu tiên chủ thể chính (vật thể trung tâm, chiếm diện tích lớn nhất hoặc chủ đề hoạt động); chọn đúng 1 nhãn từ taxonomy theo quy tắc. | Kiểm tra nhãn được gán có đúng quy tắc ưu tiên trong guideline không; kiểm tra nhãn có hợp lệ trong taxonomy không; gắn cờ/trả về rework nếu gán sai chủ thể. |
| Phát hiện vật thể | Tập hợp các hình chữ nhật bao quanh từng vật thể, mỗi hộp gồm (`class_id`, `class_name`) và 4 giá trị tọa độ hộp bao (`bbox_xyxy`: `x_min, y_min, x_max, y_max`). | Trên ảnh `kitchen`, các vật thể nhỏ/xa như thìa (`spoon`), cốc (`cup`) có score thấp (0.38 - 0.45) dễ bị sót; lò nướng và người ở mép trái bị cắt mép (`x = 0.11`, `x = 0.12`); các bát xếp chồng khó phân định ranh giới box. | Vẽ bounding box ôm khít (tight box) toàn bộ phần nhìn thấy của từng đối tượng; gán đúng lớp; gắn cờ thuộc tính nếu vật bị cắt mép/che khuất theo guideline. | Soát lỗi bỏ sót vật thể (False Negative) hoặc vẽ thừa vật nhiễu (False Positive); kiểm tra độ khít của box (không quá rộng, không cắt phạm vi vật); kiểm tra đúng class. |
| Instance segmentation | Mặt nạ điểm ảnh hoặc đa giác khép kín (`polygon_xy`) kèm `instance_id` duy nhất và (`class_id`, `class_name`) cho từng cá thể độc lập. | Trên ảnh `kitchen`, các chiếc bát chạm đè lên nhau, bàn tay chạm quầy bếp; mô hình dự đoán biên mask bị răng cưa, một số điểm lan sang nền hoặc vật thể khác; bàn `dining table` có tới 598 điểm nhưng biên dưới lộn xộn. | Sử dụng công cụ vẽ đa giác bám khít từng pixel đường viền nhìn thấy; tách riêng từng instance độc lập dù cùng class; không vẽ lấn vào nền hay phần bị che. | Phóng to (zoom-in) kiểm tra độ chính xác mức pixel của đường biên mask; đảm bảo các đối tượng chạm nhau được tách thành các instance riêng; kiểm tra tính duy nhất của `instance_id`. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Tuyệt đối không tải lên, lưu trữ hoặc công khai bất kỳ thông tin nhận dạng cá nhân nào (như họ tên, MSSV, email, số điện thoại), ảnh riêng tư cá nhân hoặc dữ liệu nội bộ/khách hàng vào Colab, repo GitHub công khai và các file báo cáo/kết quả thực hành.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Dừng ngay mọi thao tác xử lý trên dữ liệu đó, không lưu lại hay phát tán ra ngoài, và lập tức báo cáo trực tiếp cho Giảng viên hướng dẫn / Lab Coach (hoặc Mentor phụ trách lớp) để được hướng dẫn xử lý và cô lập dữ liệu an toàn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
