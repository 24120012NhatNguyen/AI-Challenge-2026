# Hướng dẫn sử dụng hệ thống tìm kiếm video AIC

Tài liệu dành cho **thí sinh sử dụng công cụ**, không phải cho lập trình viên.
Đọc mục 1 để khởi động, mục 3 để biết quy trình giải một câu KIS.

---

## 1. Khởi động hệ thống

Hệ thống gồm 3 phần chạy song song:

| Phần | Chạy ở đâu | Cổng | Vai trò |
|---|---|---|---|
| `app.py` | Kaggle (cần GPU) | 8080 | Máy tìm kiếm: text, KNN, panel, OCR/ASR, frame range |
| `socket_app.py` | Kaggle | 8081 | Lưu đáp án, ignore, export CSV/ZIP |
| Frontend Next.js | Máy cá nhân | 3000 | Giao diện |

### Bước 1 — Kéo code mới nhất về Kaggle

```bash
cd /kaggle/working/AIC2025-main
git pull
```

> Nếu notebook đã clone từ trước và `git pull` báo lỗi, xoá thư mục và clone lại.
> **Lưu ý:** thư mục `dict/` và ảnh keyframe KHÔNG nằm trong git — chúng đến từ
> Kaggle Dataset, nên xoá clone không làm mất dữ liệu.

### Bước 2 — Chạy backend trên Kaggle

Mở 2 cell riêng (hoặc 2 tiến trình nền):

```bash
python app.py          # cổng 8080
python socket_app.py   # cổng 8081
```

**Lần chạy đầu sau khi cập nhật code sẽ lâu hơn bình thường (thêm 2–5 phút).**
Đó là do hệ thống tự phát hiện và build lại các chỉ mục bị lệch dữ liệu
(ASR và tag). Trong log sẽ thấy:

```
WARNING | Ma trận ASR lệch kích thước (12844 dòng, cần 25330). Xoá cache và build lại.
WARNING | tag_embedding.bin lệch (4585 vector, cần 3163 tag ...). Build lại.
```

Đây là **bình thường và chỉ xảy ra một lần**. Các lần chạy sau sẽ nhanh như cũ.

### Bước 3 — Mở tunnel

```bash
# Backend tìm kiếm (8080) — ngrok
ngrok http 8080

# Server socket/đáp án (8081) — cloudflared
cloudflared tunnel --url http://localhost:8081
```

### Bước 4 — Cập nhật `web_url.js` (BẮT BUỘC mỗi lần tunnel đổi URL)

Mở `frontend/src/helper/web_url.js` trên máy cá nhân và dán URL mới:

```js
export const web_url    = "https://<url-ngrok>.ngrok-free.dev";        // ← app.py (8080)
export const socket_url = "https://<url-cloudflared>.trycloudflare.com"; // ← socket_app.py (8081)
export const server     = "https://<url-cloudflared>.trycloudflare.com/submit";
export const session    = "1";
```

> ⚠️ URL ngrok/cloudflared **đổi mỗi lần khởi động lại**. Quên bước này là
> nguyên nhân số 1 của lỗi "không tìm được gì" / "ảnh không hiện".

### Bước 5 — Chạy frontend trên máy cá nhân

```bash
cd frontend
npm install      # chỉ cần lần đầu
npm run dev
```

Mở http://localhost:3000

### Bước 6 — Kiểm tra nhanh trước khi thi

Mở trình duyệt vào `<web_url>/diagnostics`. Nếu mọi thứ ổn sẽ thấy `"ok": true`.
Xem chi tiết ở [mục 6](#6-endpoint-diagnostics--tự-kiểm-tra-sức-khoẻ-hệ-thống).

---

## 2. Bảng giải thích toàn bộ nút / ô trên giao diện

Trạng thái dưới đây chỉ ghi **"Hoạt động"** cho những mục đã chạy thật với dữ
liệu thật và có kết quả đúng.

### 2.1. Thanh tìm kiếm chính (giữa màn hình, phía trên)

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Ô `K` | Số keyframe tối đa trả về (mặc định 500) | — | **Hoạt động.** Xoá trắng ô sẽ tự về 500 thay vì báo lỗi |
| Ô `Type here...` | Câu truy vấn chính (gõ tiếng Việt được) | `POST /textsearch` | **Hoạt động** |
| Gợi ý dịch (dòng dưới ô tìm kiếm) | Dịch Việt→Anh; bấm để copy | `POST /translate` | **Hoạt động** |
| 🎤 (micro) | Đọc câu truy vấn bằng giọng nói | — (Web Speech API) | Chưa kiểm chứng — phụ thuộc trình duyệt |
| 🔍 (nút tròn cam) | Chạy tìm kiếm | `POST /textsearch` + `POST /getrec` | **Hoạt động** |
| `Clear` | Xoá lịch sử truy vấn và kết quả | — | **Hoạt động** |
| Tab lịch sử truy vấn | Xem lại kết quả các lần tìm trước | — (lưu trong bộ nhớ trình duyệt) | **Hoạt động** |

### 2.2. Bộ lọc và chế độ

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Dropdown `No Filter` / `Filter Forwards` / `Filter Backwards` | Tìm cảnh **sau** (Forwards) hoặc **trước** (Backwards) kết quả hiện tại — dùng cho truy vấn nhiều sự kiện nối tiếp | `POST /textsearch` (`filtervideo`) | **Hoạt động** |
| Ô `range` | Số shot lân cận được quét khi lọc (mặc định 3) | `POST /textsearch` (`range_filter`) | **Hoạt động.** Xoá trắng tự về 3 |
| Checkbox `Filter` | Bật chế độ lọc: chỉ tìm trong kết quả của lượt trước | `POST /textsearch` (`filter`, `id`) | **Hoạt động** |
| Checkbox `Ignore` | Loại các keyframe đã bị ẩn khỏi kết quả | `POST /getignore` + `/textsearch` | **Hoạt động** |
| Checkbox `AutoIgnore` | Tự ẩn cả trang khi chuyển trang | socket `ignore` | **Hoạt động** |
| Checkbox `Nomic` | Bật model Nomic (mạnh về ngữ nghĩa câu dài) | `POST /textsearch` | **Hoạt động** |
| Checkbox `CLIPv2` | Bật model CLIPv2 (mạnh về mô tả hình ảnh). Bật cả hai = cộng điểm hai model | `POST /textsearch` | **Hoạt động** |
| Ô `Space` (0–5) | Giới hạn phạm vi tìm kiếm theo nhóm video (0 = toàn bộ, 1–5 = 1/5 kho) | `POST /textsearch` (`search_space`) | **Hoạt động.** Giá trị 5 trước đây làm sập server, nay đã sửa |
| Checkbox `Feedback` | Bật chế độ đánh giá like/dislike trên ảnh | — | **Hoạt động** |
| Nút `Send` (cạnh Feedback) | Gửi phản hồi để xếp hạng lại | `POST /feedback` | **Hoạt động** |

### 2.3. Câu hỏi, đáp án và xuất file

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Ô `Questions` | Chọn/đặt tên câu hỏi đang làm. **Phải có trước khi lưu đáp án** | `POST /getquestions` | **Hoạt động** |
| Ô `Username...` + 🔒 | Tên người dùng (để biết ai làm câu nào). Bấm khoá để sửa | `POST /getquestions` | **Hoạt động** |
| Nút `CSV` | Tải file CSV của **câu hỏi đang chọn**, đúng định dạng KIS | `GET /export/kis` | **Hoạt động** |
| Nút `ZIP` | Tải `submission.zip` chứa tất cả câu tên kết thúc bằng `-kis` | `GET /export/submission_zip` | **Hoạt động** |
| Nút `View` | Mở trang xem lại toàn bộ đáp án đã đánh dấu | socket `viewsubmitted` | **Hoạt động** |

### 2.4. Các nút trên mỗi ảnh (rê chuột vào ảnh)

| Biểu tượng | Tên | Chức năng | Route backend | Trạng thái |
|---|---|---|---|---|
| 🔍 | KNN | Tìm các keyframe **giống ảnh này** | `GET /imgsearch` | **Hoạt động** |
| 🎬 | Shot | Mở tab mới xem các shot lân cận trong cùng video | `GET /getvideoshot` | **Hoạt động** |
| ⬚ | Select | Nộp trực tiếp lên server BTC | `GET /submit` (qua `server`) | Phụ thuộc server BTC — chưa kiểm chứng với server thật |
| 🗄️ | sendView | **Lưu ảnh này làm đáp án** cho câu hỏi đang chọn | socket `submit` | **Hoạt động** |
| ⛶ (góc dưới phải) | Full screen | Phóng to + xem các keyframe cùng shot | `GET /relatedimg` | **Hoạt động** |
| 👁️‍🗨️ (góc trên phải) | Ignore | Ẩn/bỏ ẩn ảnh này (bấm lần 2 để bỏ ẩn) | socket `ignore` | **Hoạt động** |
| 👁️‍🗨️ (cạnh tên video) | Ignore cả video | Ẩn toàn bộ keyframe của video đó | socket `ignore` | **Hoạt động** |
| Số ở góc trên trái ảnh | `frame_id` — **con số cần điền vào file nộp** | — | — |

> **Nếu mất kết nối tới socket server**, nút 🗄️ và 👁️‍🗨️ sẽ hiện cảnh báo
> "Mất kết nối tới socket server..." thay vì im lặng không làm gì như trước.

### 2.5. Panel (cột trái)

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Ô `Search icons` + lưới icon | Tìm icon vật thể (person, car, dog...) | — (danh sách cục bộ) | **Hoạt động** |
| Dải màu (red, white, yellow...) | Kéo khối màu vào lưới để tìm theo màu ở vị trí đó | `POST /panel` | **Hoạt động** |
| Khung lưới 7×7 trắng | Kéo–thả icon/màu vào đúng vị trí muốn tìm trong khung hình | `POST /panel` | **Hoạt động** |
| Ô `Search for tags` | Chọn tag từ từ điển 1199 tag có sẵn | `POST /panel` (`tags`) | **Hoạt động** |
| Ô `Query to get tag recommendations` + 🔍 | Gõ câu mô tả → gợi ý tag phù hợp, bấm để thêm | `POST /getrec` | **Hoạt động** (trước đây lỗi 500, đã sửa) |
| Ô `OCR` | Tìm chữ **xuất hiện trên màn hình** | `POST /panel` (`ocr`) | **Hoạt động** |
| Ô `ASR` | Tìm lời **được nói** trong video | `POST /panel` (`asr`) | **Hoạt động** (trước đây sập, đã sửa) |
| Ô `Specify maximum number of objects...` | Giới hạn số lượng vật thể, vd. `person1, car2` | `POST /panel` (`amount`) | **Hoạt động** |
| Ô `K` (trong Panel) | Số kết quả Panel trả về | `POST /panel` | **Hoạt động.** Xoá trắng tự về 500 |
| Checkbox `ID` | Chỉ tìm trong kết quả của khung tìm kiếm chính | `POST /panel` (`useid`) | **Hoạt động** |
| `Clear` / `Clear Panel` / `Clear Tags` | Xoá kết quả / xoá vật thể đã kéo / xoá tag đã chọn | — | **Hoạt động** |
| Nút `Send` (viền cam) | Chạy tìm kiếm Panel | `POST /panel` | **Hoạt động** |

> Nếu một kênh (object/OCR/ASR) gặp sự cố, Panel **vẫn trả kết quả** của các
> kênh còn lại và hiện hộp thoại cảnh báo tên kênh bị lỗi.

### 2.6. FrameRange — xem một dải frame trong 1 video

Nằm ngay dưới Panel.

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Ô `Video ID` | ID video, vd. `L21_a_V024` hoặc `L21_V024` (xem [mục 4](#4-định-dạng-id-chuẩn)) | `GET /framerange` | **Hoạt động** |
| Ô `Start` / `End` | Khoảng `frame_id` muốn xem (theo số hiển thị góc trên trái ảnh) | `GET /framerange` | **Hoạt động** |
| Ô `Tìm trong dải này...` | (Tuỳ chọn) xếp lại dải theo mức khớp với câu mô tả | `GET /framerange` (`text_query`) | **Hoạt động** |
| Nút `Xem` | Chạy | `GET /framerange` | **Hoạt động** |

Không nhập ô tìm kiếm → kết quả sắp theo `frame_id` **tăng dần** (đúng thứ tự thời gian).

### 2.7. Trang View (nút `View`)

| Tên hiển thị | Chức năng | Route backend | Trạng thái |
|---|---|---|---|
| Ô `Get Question...` + `Send` | Tải danh sách đáp án đã lưu của câu hỏi | socket `viewsubmitted` | **Hoạt động** |
| Kéo–thả ảnh | Sắp xếp lại thứ tự ưu tiên đáp án | socket `reorder` | Chưa kiểm chứng bằng thao tác chuột thật |
| `Active RE` | Xin quyền sắp xếp (tránh 2 người sửa cùng lúc) | socket `activereorder` | Chưa kiểm chứng bằng thao tác chuột thật |
| `Reorder` | Lưu thứ tự mới | socket `reorder` | Chưa kiểm chứng bằng thao tác chuột thật |
| `Download` | Tải CSV từ dữ liệu đang hiển thị trên trang | — (tạo file phía trình duyệt) | **Hoạt động**. Khuyến nghị dùng nút `CSV` ở trang chính thay thế |
| Chỉ số `x/100` | Số đáp án đã lưu / giới hạn 100 dòng của BTC | — | **Hoạt động** |
| ❌ trên ảnh | Xoá đáp án khỏi danh sách | socket `clearsubmit` | **Hoạt động** |

---

## 3. Quy trình giải một câu truy vấn KIS

### Bước 0 — Đặt tên câu hỏi ĐÚNG NGAY TỪ ĐẦU

Gõ vào ô `Questions` **đúng tên file mà BTC yêu cầu, cộng hậu tố `-kis`**.
Ví dụ BTC yêu cầu nộp `query-1-kis.csv` → đặt tên câu hỏi là `query-1-kis`.

> Nút `ZIP` **chỉ gom những câu có tên kết thúc bằng `-kis`**. Đặt sai tên là
> câu đó không vào file nộp.

Đồng thời điền `Username` (mở khoá 🔒 để sửa).

### Bước 1 — Tìm kiếm thô

1. Gõ mô tả bằng tiếng Việt vào ô tìm kiếm chính.
2. Bật **cả `Nomic` và `CLIPv2`** để cộng điểm hai model (thường cho kết quả tốt nhất).
3. Để `K` khoảng 300–500.
4. Bấm 🔍.
5. Liếc dòng dịch Việt→Anh ngay dưới ô tìm kiếm để chắc câu được hiểu đúng.

### Bước 2 — Lọc dần

Chọn một trong các cách, có thể kết hợp:

**a) Lọc bằng Panel (khi câu hỏi có chi tiết cụ thể)**
- Có chữ trên màn hình → gõ vào ô `OCR`.
- Có lời nói / là bản tin thời sự → gõ vào ô `ASR`.
- Có bố cục cụ thể ("người bên trái, xe bên phải") → kéo icon vào lưới 7×7.
- Bật checkbox `ID` để Panel chỉ lọc trong kết quả bước 1.
- Bấm `Send` trong Panel.

**b) Lọc theo trình tự thời gian (câu hỏi mô tả nhiều cảnh nối tiếp)**
- Tìm cảnh thứ nhất trước.
- Chọn `Filter Forwards`, tick `Filter`, đặt `range` = 3–5.
- Gõ mô tả cảnh thứ hai rồi bấm 🔍 → hệ thống chỉ tìm trong các shot **liền sau** cảnh thứ nhất.

**c) Loại nhiễu**
- Bấm 👁️‍🗨️ trên ảnh sai / trên tên video sai.
- Tick `Ignore` rồi tìm lại — các ảnh đã ẩn sẽ không xuất hiện nữa.

### Bước 3 — Xem dải frame để xác định frame chính xác (FrameRange)

Khi đã khoanh được đúng video nhưng chưa chắc frame nào:

1. Rê chuột lên ảnh, đọc **tên video** ở cột trái (vd. `L21_V024`) và **frame_id** ở góc trên trái ảnh (vd. `1200`).
2. Điền vào FrameRange: `Video ID` = `L21_V024`, `Start` = `1000`, `End` = `1500`.
3. Bấm `Xem` → thấy toàn bộ keyframe trong dải, sắp theo thời gian.
4. Muốn thu hẹp hơn: gõ mô tả vào ô `Tìm trong dải này...` để xếp lại theo độ khớp.

### Bước 4 — Đánh dấu đáp án

Rê chuột lên ảnh đúng → bấm **🗄️ (sendView)**.
Lặp lại cho các ứng viên khác, **ưu tiên cao đứng trước** (tối đa 100).

Kiểm tra lại: bấm `View` → chọn câu hỏi → `Send`. Xoá ảnh sai bằng ❌.

### Bước 5 — Xuất CSV

Về trang chính, đảm bảo ô `Questions` đang là câu cần xuất → bấm **`CSV`**.

File tải về có dạng (không có dòng tiêu đề):

```
L21_V001,0
L24_V031,4885
L30_V079,668
```

Mỗi dòng = `<tên_video>,<frame_id>`.

### Bước 6 — Đóng gói ZIP

Bấm **`ZIP`** → tải `submission.zip`:

```
submission/query-1-kis.csv
submission/query-2-kis.csv
```

Câu nào vượt 100 dòng sẽ **bị bỏ qua** và ghi lý do vào `submission/_WARNINGS.txt`
trong zip. Luôn mở `_WARNINGS.txt` kiểm tra trước khi nộp.

### Bước 7 — Nộp lên hệ thống BTC

Tải `submission.zip` lên trang nộp bài của BTC theo hướng dẫn của cuộc thi.

---

## 4. Định dạng ID chuẩn

**Cách lấy ID đúng:**

- **Tên video**: nhìn cột dọc màu cam bên trái mỗi hàng ảnh — đó là tên video,
  dạng `L21_V024`. Đây chính là chuỗi cần điền vào cột đầu của file nộp.
- **frame_id**: con số ở **góc trên bên trái mỗi ảnh** (vd. `4885`). Đây là cột
  thứ hai của file nộp.

**Cấu trúc:**

```
L21_V024
│   └── V024  = số thứ tự video trong bộ
└── L21       = bộ dữ liệu (L21…L30)
```

Ảnh được lưu theo lô tải về nên đường dẫn có thêm chữ cái lô:
`/static/images/Keyframes/L21_a/V024/001200.jpg` → chữ `_a` **chỉ là tên thư
mục tải về**, không thuộc tên video.

Ô `Video ID` của FrameRange chấp nhận **cả hai dạng** (`L21_V024` và
`L21_a_V024`, không phân biệt hoa thường), nhưng **file nộp luôn dùng dạng
chuẩn `L21_V024`** — hệ thống tự xuất đúng, bạn không phải sửa tay.

---

## 5. Xử lý sự cố thường gặp

### "Textsearch Fetch Failed!" / bấm tìm kiếm không có gì xảy ra
Tunnel đã chết hoặc URL trong `web_url.js` đã cũ.
1. Kiểm tra tiến trình `ngrok` / `cloudflared` trên Kaggle còn sống không.
2. Nếu chết: chạy lại, **copy URL mới**, sửa `web_url.js`, lưu file (Next.js tự nạp lại).

### "Mất kết nối tới socket server..."
Tunnel cloudflared (cổng 8081) đã chết. Mở lại và cập nhật `socket_url` **và**
`server` trong `web_url.js`. **Đáp án bấm 🗄️ lúc đó chưa được lưu — bấm lại sau khi kết nối lại.**

### Ảnh không hiện (ô xám / vỡ ảnh)
Ảnh keyframe được phục vụ từ **máy cá nhân**, không phải Kaggle. Kiểm tra:
```
frontend/public/static/images/Keyframes/<lô>/<video>/<frame>.jpg
```
vd. `frontend/public/static/images/Keyframes/L21_a/V024/001200.jpg`.
Thiếu thư mục này thì phải tải lại dataset keyframes.

### "Input should be a valid integer"
Đã sửa — ô số để trống nay tự lùi về mặc định (K=500, range=3, Space=0).
Nếu vẫn gặp: bạn đang chạy code cũ trên Kaggle → `git pull` rồi khởi động lại backend.

### "Video 'XXX' không tồn tại" ở FrameRange
Gõ sai tên video. Copy đúng chuỗi ở **cột cam bên trái hàng ảnh** (vd. `L21_V024`).
Thông báo lỗi cũng nhắc lại định dạng hợp lệ.

### Panel hiện "Panel search cảnh báo: asr: ..."
Một kênh gặp sự cố nhưng các kênh còn lại vẫn chạy — kết quả hiển thị là của
các kênh còn lại. Xem log backend trên Kaggle để biết chi tiết.

### Vòng xoay loading quay mãi không dừng
Đã sửa ở các nút KNN, Feedback và Panel Send: khi request lỗi sẽ hiện thông báo
và tắt vòng xoay. Nếu vẫn gặp, tải lại trang (F5) và kiểm tra tunnel.

### Nút ZIP báo "Không có câu hỏi nào kết thúc bằng '-kis'"
Tên câu hỏi đặt sai. Đổi tên câu hỏi thành `<tên-BTC-yêu-cầu>-kis` rồi đánh dấu lại đáp án.

### Backend khởi động rất lâu ở lần chạy đầu
Bình thường — hệ thống đang build lại chỉ mục ASR/tag bị lệch. Chỉ xảy ra một lần.

---

## 6. Endpoint `/diagnostics` — tự kiểm tra sức khoẻ hệ thống

Mở trình duyệt vào:

```
<web_url>/diagnostics
```

(vd. `https://abcd-1234.ngrok-free.dev/diagnostics`)

Kết quả mong đợi:

```json
{
  "ok": true,
  "n_keyframes": 97358,
  "n_videos": 785,
  "asr_dir": ".../dict/audio_ARS",
  "example_video_id": "L21_V001",
  "checks": [
    {"name": "ocr_matrix",  "ok": true, "detail": "97358 dòng / 97358 keyframe"},
    {"name": "asr_matrix",  "ok": true, "detail": "25330 dòng / 25330 đoạn ASR"},
    {"name": "faiss_nomic", "ok": true, "detail": "97358 vector / 97358 keyframe"},
    ...
  ]
}
```

**Cách đọc:**

- `"ok": true` → mọi chỉ mục khớp dữ liệu, yên tâm thi.
- `"ok": false` → tìm mục nào có `"ok": false` trong `checks`:

| Mục lỗi | Hậu quả | Cách xử lý |
|---|---|---|
| `asr_matrix` | Ô ASR không dùng được | Xoá `dict/bin/audio_bin/` rồi khởi động lại `app.py` |
| `ocr_matrix` | Ô OCR không dùng được | Xoá `dict/bin/ocr_bin/` rồi khởi động lại |
| `object_matrix_*` | Panel kéo icon/màu không dùng được | Xoá `dict/bin/contexts_bin/` rồi khởi động lại |
| `faiss_nomic` / `faiss_clipv2` | Tìm kiếm text sai lệch | Dataset `dict/` không khớp bộ ảnh — tải lại dataset |
| `video_division` | Ô `Space` 1–5 thiếu video | Tải lại `dict/video_division_tag.json` |

`example_video_id` cho biết **định dạng tên video chuẩn** mà hệ thống đang dùng —
tiện để đối chiếu khi FrameRange báo không tìm thấy video.

> Nên gọi `/diagnostics` một lần **ngay sau khi khởi động backend, trước mỗi buổi thi**.
