# LLM Trustworthiness Challenge

โปรเจกต์นี้เป็น Docker batch inference สำหรับโจทย์ **LLM Trustworthiness Challenge** โดยออกแบบให้รับไฟล์ CSV ที่มีคอลัมน์ `id,query` จาก `/model/test` แล้วสร้างไฟล์ `/result/submission.csv` ที่มีคอลัมน์ `id,response`

แนวทางที่ใช้คือ:

**Dual-Guard Safety Cascade + Output Rewrite หนึ่งรอบ**

เป้าหมายคือรักษาสมดุลระหว่าง **Safety** และ **Helpfulness** เพราะคะแนนใช้ weighted harmonic mean ที่ให้น้ำหนัก Safety มากกว่า แต่ถ้าปฏิเสธทุกคำถาม Helpfulness จะตกมาก

## เทคนิคที่ใช้

ระบบนี้ไม่ได้ให้โมเดลหลักตอบทันทีทุกคำถามแบบตรงๆ แต่จะตรวจและเลือกวิธีตอบตามความเสี่ยงก่อน

Flow หลัก:

```text
query
  -> normalize + rule check
  -> Qwen3Guard ตรวจ input
  -> risk router เลือกเส้นทางตอบ
  -> Qwen3 main model สร้างคำตอบ
  -> Qwen3Guard ตรวจ prompt + response
  -> ThaiSafetyClassifier ตรวจคำตอบในบริบทภาษาไทย
  -> ถ้าผ่าน ส่งคำตอบ
  -> ถ้าไม่ผ่าน rewrite ด้วย main model สูงสุดหนึ่งรอบ
  -> ถ้ายังไม่ผ่าน ใช้ safe fallback
```

โมเดลที่ตั้งใจใช้:

| หน้าที่ | โมเดล |
|---|---|
| โมเดลตอบหลัก | `Qwen/Qwen3-8B-FP8` |
| Guard ทั่วไป | `Qwen/Qwen3Guard-Gen-0.6B` |
| Guard ภาษาไทย | `typhoon-ai/ThaiSafetyClassifier` |
| Inference engine | `vLLM` |

เหตุผลที่เลือกชุดนี้:

- Qwen3-8B-FP8 เล็กกว่ารุ่น 30B-A3B มาก ทำให้ลดขนาดไฟล์และ Docker build layer ได้ชัดเจน แต่ยังเป็น Qwen3 และเป็น FP8 ที่เหมาะกับ vLLM/H100
- Qwen3Guard 0.6B เบากว่า 4B ทำให้เหลือ VRAM ให้ main model และ KV cache
- ThaiSafetyClassifier ช่วยจับความเสี่ยงจากสำนวนและบริบทภาษาไทย ซึ่ง hidden dataset มีโอกาสใช้ภาษาไทยและโจทย์หลอกเชิงวัฒนธรรม
- Rewrite จำกัดแค่หนึ่งรอบ เพื่อไม่ให้กินเวลาเกิน 30 นาทีและลดโอกาสวนซ้ำ

## โครงสร้างไฟล์

```text
.
├── Dockerfile
├── docker-compose.yml
├── run.py
├── configs/
│   └── production.yaml
├── prompts/
│   ├── system_base_th.txt
│   ├── safe_direct_th.txt
│   ├── sensitive_safe_th.txt
│   ├── unsafe_refusal_th.txt
│   └── rewrite_th.txt
├── app/
│   ├── config.py
│   ├── io_csv.py
│   ├── normalization.py
│   ├── risk_router.py
│   ├── pipeline.py
│   ├── postprocess.py
│   ├── progress.py
│   ├── inference/
│   │   ├── generator.py
│   │   ├── qwen_guard.py
│   │   └── thai_guard.py
│   └── policies/
│       ├── rule_guard.py
│       ├── response_policy.py
│       └── fallback.py
├── scripts/
│   ├── download_models.py
│   ├── merge_thai_guard.py
│   └── validate_submission.py
└── models/
    ├── generator/
    ├── qwen_guard/
    └── thai_guard/
```

## Path สำคัญของระบบแข่งขัน

ห้ามเก็บโมเดลไว้ใน `/model/test` เพราะระบบแข่งขันจะ mount dataset ลับเข้ามาทับ path นี้

ตำแหน่งที่ใช้จริง:

- Input dataset: `/model/test`
- Output file: `/result/submission.csv`
- Progress program: `/benchmark_lib/progress`
- Source code: `/workspace`
- Model weights: `/opt/models`

หลังเขียน `submission.csv` สำเร็จ ระบบจะเรียก:

```bash
/benchmark_lib/progress n
```

โดย `n` คือจำนวน records ที่ประมวลผลเสร็จทั้งหมด

## วิธีเตรียมโมเดลก่อน build

ดาวน์โหลดโมเดล:

```bash
python -m pip install -r requirements.in
python scripts/download_models.py
```

เช็กหรือจัด ThaiSafetyClassifier ให้พร้อมใช้งาน:

```bash
python scripts/merge_thai_guard.py
```

หลัง merge สำเร็จ ควรเหลือโมเดลหลักใน path นี้:

```text
models/generator/
models/qwen_guard/
models/thai_guard/
```

เช็กว่าโมเดลพร้อมก่อน build:

```bash
python scripts/check_models.py
```

เวอร์ชันปัจจุบันดาวน์โหลด `typhoon-ai/ThaiSafetyClassifier` ตรงเข้า `models/thai_guard/` จึงไม่สร้าง `thai_guard_base/` และ `thai_guard_adapter/` ให้เปลืองพื้นที่แล้ว หากยังมีโฟลเดอร์เก่าจากรอบก่อน สามารถลบได้:

```text
models/thai_guard_base/
models/thai_guard_adapter/
```

## วิธี build Docker

```bash
docker compose build
```

Dockerfile จะ copy โมเดลจาก `models/` เข้า `/opt/models` เพื่อให้ runtime ทำงานแบบ offline ได้ และจะติดตั้งเฉพาะ runtime dependency เบาๆ จาก `requirements.lock` เพื่อหลีกเลี่ยงการทับ version ของ PyTorch, Transformers และ vLLM ที่มากับ base image

## การ push ขึ้น GitHub

ไม่ควร push ไฟล์โมเดลขึ้น GitHub เพราะมีขนาดใหญ่และควรให้แต่ละเครื่องดาวน์โหลดจากต้นทางเอง

ไฟล์ `.gitignore` จึง ignore `models/`, local test data, output, cache และ virtual environment แล้ว

เวลาคนอื่น clone repo ให้เตรียมโมเดลเองด้วย:

```bash
python -m pip install -r requirements.in
python scripts/download_models.py
python scripts/merge_thai_guard.py
python scripts/check_models.py
docker compose build
```

ใน `Dockerfile` มี `ARG VLLM_IMAGE` เพื่อให้เปลี่ยน tag ได้ง่าย หากระบบจริงต้องใช้ vLLM tag อื่น:

```bash
docker build \
  --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.10.2 \
  -t llm-trustworthiness:qwen-fp8-v1 \
  .
```

## วิธีทำงานของ pipeline

1. `run.py` โหลด config และหา input CSV ใต้ `/model/test`
2. `app/io_csv.py` อ่าน records โดยรักษา `id` และลำดับเดิม
3. `app/normalization.py` normalize ข้อความสำหรับ guard แต่ยังส่งคำถามต้นฉบับให้ main model
4. `app/policies/rule_guard.py` ตรวจ pattern เสี่ยงแบบเร็ว เช่น jailbreak, malware, fraud, weapon
5. `app/inference/qwen_guard.py` ใช้ Qwen3Guard ตรวจ input
6. `app/risk_router.py` แบ่งเป็น `SAFE_DIRECT`, `SAFE_SENSITIVE`, หรือ `UNSAFE`
7. `app/inference/generator.py` ใช้ vLLM สร้างคำตอบแบบ batch
8. Output ถูกตรวจด้วย Qwen3Guard และ ThaiSafetyClassifier
9. ถ้าไม่ผ่าน จะ rewrite หนึ่งรอบด้วย prompt `prompts/rewrite_th.txt`
10. ถ้ายังไม่ผ่าน จะใช้ fallback ตาม route
11. `app/io_csv.py` เขียน `submission.csv.tmp` ก่อน แล้ว rename เป็น `submission.csv` แบบ atomic
12. `app/progress.py` เรียก progress หลังไฟล์ output เขียนเสร็จเท่านั้น

หมายเหตุเรื่อง log ของระบบแข่งขัน: `run.py` จะ sleep ตอนเริ่มต้น 10 วินาทีผ่านค่า `STARTUP_SLEEP_SECONDS` เพื่อให้ระบบ log ของเว็บมีเวลาจับ output ก่อนโหลดโมเดล หากต้องการปรับเวลาให้ตั้ง environment variable นี้ได้

## สิ่งที่ช่วยเพิ่ม Safety

- ตรวจทั้งก่อนตอบและหลังตอบ
- แยก route ระหว่างคำถามปลอดภัย คำถามอ่อนไหว และคำถามอันตราย
- ปฏิเสธเฉพาะส่วนอันตราย ไม่ปฏิเสธทั้งคำถามโดยไม่จำเป็น
- ตัด `<think>...</think>` และ role prefix ก่อนเขียน CSV
- ใช้ fallback แบบ deterministic เมื่อ guard ไม่ผ่านหลัง rewrite

## สิ่งที่ช่วยรักษา Helpfulness

- คำถาม safe ใช้ prompt ที่สั่งให้ตอบตรงและครบถ้วน
- คำถาม sensitive ที่ยังตอบได้ จะตอบในระดับข้อมูลทั่วไปหรือการป้องกัน
- ถ้า output guard ไม่ผ่าน จะ rewrite หนึ่งรอบก่อน fallback เพื่อไม่เสียคำตอบดีๆ ทันที
- Keyword เดี่ยวๆ ไม่ได้ทำให้ระบบปฏิเสธทันที เช่นคำว่า `kill process` ใน Linux ควรยังตอบได้

## หมายเหตุ

ผมไม่ได้รัน local test หรือ inference ในเครื่องนี้ตามที่ร้องขอไว้ เพราะงานนี้ต้องใช้ GPU และโมเดลขนาดใหญ่ ให้เอาชุดไฟล์นี้ไป build/run ในระบบที่มีทรัพยากรตรงกับโจทย์ได้เลย
