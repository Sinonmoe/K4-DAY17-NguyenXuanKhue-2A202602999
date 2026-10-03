# BÁO CÁO PHÂN TÍCH HIỆU NĂNG VÀ HÀNH VI HỆ THỐNG MEMORY (DAY 17)

Báo cáo này đối chiếu kết quả đo lường thực nghiệm từ hai bộ benchmark chuẩn hóa (`data/conversations.json` và `data/advanced_long_context.json`) nhằm phân tích sâu sắc các trade-off kỹ thuật giữa **Baseline Agent** (chỉ có short-term memory) và **Advanced Agent** (tích hợp short-term, persistent `User.md` và compact memory).

---

## 1. Kết quả Benchmark thực nghiệm

Toàn bộ dữ liệu được thu thập trên trạng thái sạch (`state/` được làm mới trước khi chạy) với lệnh:
```bash
python src/benchmark.py
```

### Bảng 1: Standard Benchmark (`data/conversations.json`)
*Quy mô: 10 hội thoại, ~10 lượt/hội thoại, user `dungct`, 14 câu hỏi cross-session recall.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,507 | 14,700 | 0.00 | 0.30 | 0 | 0 |
| **Advanced** | 2,836 | 26,539 | **1.00** | **1.00** | **269** | 0 |

---

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json`)
*Quy mô: 1 hội thoại 16 lượt chứa ngữ cảnh tin tức kỹ thuật rất dài, user `dungct_stress`, 3 câu hỏi cross-session recall.*

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 255 | 22,198 | 0.00 | 0.30 | 0 | 0 |
| **Advanced** | 704 | **10,064** | **1.00** | **1.00** | **190** | **7** |

---

## 2. Trả lời bốn câu hỏi trọng tâm (Guide.md - Bước 8)

### 2.1. Vì sao Advanced có recall tốt hơn Baseline?
- **Số liệu chứng minh**: 
  - Tại cả 2 bảng, `Cross-session recall` của Baseline đều đạt **`0.00`**, trong khi Advanced đạt điểm tuyệt đối **`1.00`** (14/14 câu ở Standard và 3/3 câu ở Stress).
  - Cột `Memory growth (bytes)` của Baseline bằng **`0`**, trong khi Advanced tăng **`269 bytes`** ở Standard và **`190 bytes`** ở Stress.
- **Cơ chế trong mã nguồn**:
  - **Baseline** chỉ duy trì `SessionState` khóa theo `thread_id` (`self.sessions: dict[str, SessionState]`). Khi câu hỏi recall được gửi trong một thread mới (`<conv_id>_recall`), Baseline không tìm thấy lịch sử nào và buộc phải trả lời *"Tôi không có thông tin về bạn trong phiên trò chuyện mới này."*.
  - **Advanced** có đường dẫn fact độc lập:
    1. Khi nhận tin nhắn người dùng, `extract_profile_updates()` trích xuất các facts ổn định (tên, nghề nghiệp, nơi ở, đồ uống...).
    2. Các facts này được ghi xuống đĩa bền vững qua `profile_store.upsert_fact(user_id, k, v)` tại đường dẫn `state/profiles/<user_id>/User.md`.
    3. Khi thread mới mở ra để hỏi recall, `_offline_response()` trực tiếp đọc file `User.md` thông qua `profile_store.facts(user_id)` và truy xuất câu trả lời chính xác 100%.

---

### 2.2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
- **Số liệu chứng minh**:
  - Tại bảng Standard Benchmark (các hội thoại ngắn ~10 lượt), `Prompt tokens processed` của Advanced là **26,539 tokens**, cao hơn gần gấp đôi so với Baseline (**14,700 tokens**).
  - Cột `Compactions` ở cả hai agent đều bằng **`0`**.
- **Cơ chế trong mã nguồn**:
  - Ở hội thoại ngắn, độ dài ngữ cảnh chưa từng vượt qua ngưỡng kích hoạt compact (`compact_threshold_tokens = 800`). Do đó, compact memory chưa từng nén một lượt nào (`Compactions = 0`).
  - Trong khi đó, ở mỗi lượt hội thoại của Advanced, hàm `_estimate_prompt_context_tokens()` tính toán tổng tải ngữ cảnh gồm: `User.md + summary + recent_messages`. Việc luôn mang theo tệp hồ sơ `User.md` (~269 bytes $\approx$ 65 tokens) ở mỗi lượt chat đã tạo ra một khoản chi phí cố định (overhead).
  - **Kết luận đánh đổi**: Với hội thoại ngắn, chi phí token của Advanced cao hơn là sự đánh đổi tất yếu và hợp lý để sở hữu khả năng nhớ xuyên phiên (*cross-session persistence*).

---

### 2.3. Vì sao Compact Memory có lợi thế vượt trội ở hội thoại dài?
- **Số liệu chứng minh**:
  - Tại bảng Stress Benchmark (chuỗi 16 lượt rất dài), `Prompt tokens processed` của Baseline lên tới **22,198 tokens**, trong khi Advanced chỉ tiêu tốn **10,064 tokens** — **tiết kiệm hơn 54.6% tải ngữ cảnh**.
  - `Compactions` của Advanced đạt **7 lần**, trong khi Baseline bằng **0**.
- **Cơ chế trong mã nguồn**:
  - **Baseline không nén**: Qua 16 lượt dài, mỗi lượt Baseline phải tải toàn bộ $k$ tin nhắn trước đó vào prompt. Tải ngữ cảnh tăng theo cấp số cộng của độ dài hội thoại, dẫn đến tổng chi phí prompt tích lũy tăng phi mã ($O(N^2)$).
  - **Advanced nén chủ động**: Khi tổng token trong thread vượt ngưỡng 800, `CompactMemoryManager.append()` kích hoạt:
    1. Cắt các tin nhắn cũ (`messages[:-keep_messages]`) đưa vào `summarize_messages()`.
    2. Chỉ giữ lại đúng `keep_messages = 4` tin nhắn gần nhất nguyên văn.
    3. Thay thế hàng ngàn ký tự văn bản cũ bằng một đoạn tóm tắt ngắn gọn (~80 tokens).
  - **Điểm mấu chốt**: Compact memory **chủ yếu tối ưu cột `Prompt tokens processed`** (lượng ngữ cảnh phải mang theo qua các lượt), chứ không phải `Agent tokens only` (token do model sinh ra để trả lời).

---

### 2.4. Tăng trưởng file Memory và các rủi ro hệ thống
- **Số liệu chứng minh**:
  - `Memory growth (bytes)` tăng 269 bytes ở Standard và 190 bytes ở Stress.
  - Sau 10 phiên hội thoại, file `User.md` của user `dungct` đạt kích thước ổn định khoảng 300–400 bytes.
- **Rủi ro thực tế khi vận hành**:
  1. **Rủi ro phình to file (Unbounded Growth)**: Nếu người dùng trò chuyện hàng trăm phiên và hệ thống trích xuất mọi sở thích vụn vặt, `User.md` sẽ tăng lên hàng chục KB. Khi đó, chi phí overhead chèn `User.md` vào prompt của mỗi lượt sẽ triệt tiêu lợi ích tiết kiệm của compact memory.
  2. **Rủi ro lưu sai Fact do Nhiễu (Noise Pollution)**: Người dùng có thể đùa cợt (*"chắc chuyển sang làm product manager"*), đi công tác ngắn ngày (*"bay ra Hà Nội họp 2 ngày"*), hoặc chỉ đặt câu hỏi (*"mình tên gì?"*). Nếu bộ trích xuất quá tham, nó sẽ ghi đè fact sai vào `User.md` và làm hỏng toàn bộ recall.
  3. **Rủi ro mất chi tiết do nén (Lossy Compression)**: `summarize_messages()` tóm tắt lược bỏ các số liệu kỹ thuật chi tiết để tiết kiệm token. Nếu người dùng hỏi lại một thông số vụn vặt từ 15 lượt trước không nằm trong `User.md`, thông tin đó có thể đã biến mất vĩnh viễn.

---

## 3. Phép kiểm chứng tắt Compact Memory (Ablation Study)

Để chứng minh độc lập vai trò của lớp compact memory, chúng tôi thực hiện thí nghiệm tắt compact bằng cách đặt `compact_threshold_tokens = 999,999` trên bộ Stress test:

| Cấu hình | Prompt tokens processed | Cross-session recall | Compactions |
| :--- | :---: | :---: | :---: |
| **Baseline (Không memory)** | 22,198 | 0.00 | 0 |
| **Advanced (Tắt Compact)** | 26,495 | 1.00 | 0 |
| **Advanced (Bật Compact - Mặc định)** | **10,064** | **1.00** | **7** |

**Nhận định then chốt**:
- Khi tắt compact, Advanced tốn nhiều prompt tokens nhất (26,495 tokens) vì phải kéo theo toàn bộ lịch sử nguyên văn cộng thêm `User.md`.
- Khi bật compact, prompt tokens giảm ngay **62%** (từ 26,495 xuống 10,064).
- Thí nghiệm chứng minh rõ: **`User.md` mang lại recall, còn `Compact memory` mang lại hiệu quả chi phí token**.

---

## 4. Phần Bonus mở rộng (Mốc 90–100 điểm)

Chúng tôi đã thiết kế và triển khai hai cơ chế mở rộng kỹ thuật gắn liền với các điểm yếu quan sát được:

### 4.1. Hướng Bonus lựa chọn:
1. **Confidence Thresholding & Intent Filtering** (Ngưỡng tin cậy và phân loại ý định).
2. **Conflict Handling & Atomic Correction** (Xử lý xung đột và cập nhật nguyên tử khi có đính chính).

### 4.2. Trả lời ba tiêu chí bắt buộc của Rubric.md:

#### 1. Bonus giải quyết vấn đề gì?
- **Ngăn chặn ô nhiễm dữ liệu từ câu hỏi và tin đồn/nhiễu**: Người dùng thường xuyên hỏi lại thông tin (*"Bạn có thể nhắc lại tên mình không?"*) hoặc nhắc đến địa điểm/nghề nghiệp trong câu đùa/sự kiện tạm thời (*"đùa chuyển sang product manager"*, *"bay ra Hà Nội họp"*). Nếu không có ngưỡng lọc, agent sẽ nhận nhầm các thực thể này là facts mới và ghi đè vào `User.md`.
- **Giải quyết triệt để mâu thuẫn fact (Conflict)**: Khi người dùng đính chính nơi ở từ *Huế* sang *Đà Nẵng*, hệ thống cũ nếu chỉ append sẽ để tồn tại cả hai địa điểm trong file, dẫn đến việc model bối rối khi trả lời câu hỏi *"Nơi ở hiện tại của mình là đâu?"*.

#### 2. Nó cải thiện Recall và Token Cost như thế nào?
- **Cải thiện Recall**: Nhờ `Conflict Handling` với cơ chế `edit_text()` in-place, fact cũ bị xóa bỏ hoàn toàn ngay khi fact mới xuất hiện. Khi được hỏi câu bẫy (*"Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nơi ở và nghề nghiệp hiện tại?"*), agent trả lời chính xác 100% (*MLOps engineer tại Đà Nẵng*), đạt Recall tuyệt đối `1.00`.
- **Tối ưu Token Cost & File Size**: Nhờ `Confidence Thresholding` lọc bỏ toàn bộ câu hỏi và thông tin gây nhiễu, file `User.md` chỉ tăng đúng **190–269 bytes** thay vì phình to thành hàng nghìn bytes qua mỗi lượt hội thoại.

#### 3. Nó tạo thêm rủi ro (Risk / Trade-off) gì cho hệ thống?
- **Rủi ro bỏ sót fact hợp lệ (False Negatives)**: Đặt ngưỡng tin cậy quá khắt khe có thể khiến agent bỏ qua các câu nói tự nhiên, thiếu từ khóa khẳng định của người dùng (ví dụ: *"tuần sau mình chuyển vào Sài Gòn sống"* có thể bị coi là kế hoạch tạm thời nếu không có pattern nhận diện).
- **Rủi ro mất dữ liệu lịch sử (Irreversible Overwrite)**: Khi giải quyết xung đột bằng cách ghi đè nguyên tử, fact cũ bị xóa vĩnh viễn khỏi `User.md`. Nếu người dùng hỏi lại theo dạng lịch sử (*"Trước khi ở Đà Nẵng mình từng ở đâu?"*), agent sẽ không thể trả lời được nếu thông tin đó cũng đã bị compact memory nén mất trong thread.

---

## 5. Quy ước cấu hình và Môi trường tái lập (Reproducibility)

Để người chấm hoặc lab coach có thể chạy lại và tái lập chính xác 100% kết quả trên:

### Cài đặt môi trường:
```bash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

### Các biến môi trường hỗ trợ (trong file `.env`):
- `LLM_PROVIDER`: Provider chính (`openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`).
- `LLM_MODEL`: Tên model (mặc định `gpt-4o-mini`).
- `COMPACT_THRESHOLD_TOKENS`: Ngưỡng nén token (mặc định `800`).
- `COMPACT_KEEP_MESSAGES`: Số tin nhắn giữ nguyên văn sau nén (mặc định `4`).
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `CUSTOM_BASE_URL`.

### Lệnh chạy benchmark và kiểm tra:
```bash
# Xóa trạng thái sạch và chạy benchmark so sánh hai bộ dữ liệu
if (Test-Path state) { Remove-Item -Recurse -Force state }
python src/benchmark.py
```
