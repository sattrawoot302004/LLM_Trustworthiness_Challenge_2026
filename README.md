# LLM Trustworthiness Challenge

โปรเจกต์นี้เป็น Docker batch inference สำหรับโจทย์ **LLM Trustworthiness Challenge** โดยออกแบบให้รับไฟล์ CSV ที่มีคอลัมน์ `id,query` จาก `/model/test` แล้วสร้างไฟล์ `/result/submission.csv` ที่มีคอลัมน์ `id,response`

แนวทางที่ใช้คือ:

**Rule-Routed Main Model + ThaiSafety Output Rewrite หนึ่งรอบ**

เป้าหมายคือรักษาสมดุลระหว่าง **Safety** และ **Helpfulness** เพราะคะแนนใช้ weighted harmonic mean ที่ให้น้ำหนัก Safety มากกว่า แต่ถ้าปฏิเสธทุกคำถาม Helpfulness จะตกมาก

## เทคนิคที่ใช้

ระบบนี้ไม่ได้ให้โมเดลหลักตอบทันทีทุกคำถามแบบตรงๆ แต่จะตรวจและเลือกวิธีตอบตามความเสี่ยงก่อน

Flow หลัก:

```text
query
  -> normalize + rule check
  -> risk router เลือกเส้นทางตอบ
  -> Qwen3 main model สร้างคำตอบ
  -> response rule check
  -> ThaiSafetyClassifier ตรวจคำตอบในบริบทภาษาไทย
  -> ถ้าผ่าน ส่งคำตอบ
  -> ถ้าไม่ผ่าน rewrite ด้วย main model สูงสุดหนึ่งรอบ
  -> ThaiSafetyClassifier ตรวจซ้ำ
  -> ถ้ายังไม่ผ่าน ใช้ contextual safe fallback
```

โมเดลที่ตั้งใจใช้:

| หน้าที่ | โมเดล |
|---|---|
| โมเดลตอบหลัก | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` |
| Guard ภาษาไทย | `typhoon-ai/ThaiSafetyClassifier` |
| Inference engine | `vLLM` |

เหตุผลที่เลือกชุดนี้:

- Qwen3-30B-A3B-Instruct-2507-FP8 เป็น checkpoint FP8 official ของ Qwen เหมาะกับ H100 40 GB และ vLLM มากกว่าการใช้ BF16 เต็มรูปแบบ
- ไม่โหลด Qwen3Guard เพิ่ม เพื่อคืน VRAM และเวลา inference ให้ main model กับ KV cache
- ThaiSafetyClassifier ช่วยจับความเสี่ยงจากสำนวนและบริบทภาษาไทย ซึ่ง hidden dataset มีโอกาสใช้ภาษาไทยและโจทย์หลอกเชิงวัฒนธรรม
- ThaiSafetyClassifier ตั้งค่าแบบใกล้ evaluator มากขึ้น โดยใช้รูปแบบ `input: ... output: ...`, `max_length=128` และ threshold 0.50 เพื่อให้ rewrite เกิดกับเคสเสี่ยงจริง
- ลด `gpu_memory_utilization`, `max_num_seqs` และ guard batch size เพื่อกัน CUDA OOM ระหว่าง rewrite บน GPU เดียว
- Rewrite จำกัดแค่หนึ่งรอบ เพื่อไม่ให้กินเวลาเกิน 30 นาทีและลดโอกาสวนซ้ำ
- Fallback ไม่ใช่คำตอบปฏิเสธแบบเดียวทุกข้อ แต่เลือก template จาก route และเนื้อคำถาม เพื่อให้ยังได้คะแนน Helpfulness เท่าที่ปลอดภัย

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
│   │   └── thai_guard.py
│   └── policies/
│       ├── rule_guard.py
│       ├── response_policy.py
│       └── fallback.py
├── scripts/
│   ├── download_models.py
│   ├── merge_thai_guard.py
│   └── validate_submission.py
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

## วิธีเตรียมโมเดล

เวอร์ชันนี้ใช้วิธี **ดาวน์โหลดโมเดลตอน Docker build** ไม่ต้องเก็บ `models/` ไว้ในเครื่องและไม่ต้อง push model ขึ้น GitHub

ตอน build, Dockerfile จะรัน:

```bash
MODEL_DOWNLOAD_DIR=/opt/models python3 /workspace/scripts/download_models.py
```

แล้วดาวน์โหลดโมเดลเข้า image ที่:

```text
/opt/models/generator/
/opt/models/thai_guard/
```

จากนั้น runtime จะทำงานแบบ offline ด้วย environment `HF_HUB_OFFLINE=1` และ `TRANSFORMERS_OFFLINE=1`

## วิธี build Docker

```bash
docker compose build
```

Dockerfile จะดาวน์โหลดโมเดลเข้า `/opt/models` ระหว่าง build เพื่อให้ runtime ทำงานแบบ offline ได้ และจะติดตั้งเฉพาะ dependency เบาๆ จาก `requirements.lock` เพื่อหลีกเลี่ยงการทับ version ของ PyTorch, Transformers และ vLLM ที่มากับ base image

## การ push ขึ้น GitHub

ไม่ควร push ไฟล์โมเดลขึ้น GitHub เพราะมีขนาดใหญ่และควรให้แต่ละเครื่องดาวน์โหลดจากต้นทางเอง

ไฟล์ `.gitignore` จึง ignore `models/`, local test data, output, cache และ virtual environment แล้ว

เวลาคนอื่น clone repo สามารถ build ได้เลยถ้าเครื่อง build มี internet:

```bash
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
3. `app/normalization.py` normalize ข้อความสำหรับ rule guard แต่ยังส่งคำถามต้นฉบับให้ main model
4. `app/policies/rule_guard.py` ตรวจ pattern เสี่ยงแบบเร็ว เช่น jailbreak, cyber abuse, fraud, weapon, privacy และ obfuscation บางรูปแบบ
5. `app/risk_router.py` แบ่งเป็น `SAFE_DIRECT`, `SAFE_SENSITIVE`, หรือ `UNSAFE`
6. `app/inference/generator.py` ใช้ vLLM สร้างคำตอบแบบ batch
7. Output ถูกตรวจด้วย response rule check และ ThaiSafetyClassifier
8. ถ้าไม่ผ่าน จะ rewrite หนึ่งรอบด้วย prompt `prompts/rewrite_th.txt`
9. คำตอบ rewrite ถูกตรวจซ้ำด้วย ThaiSafetyClassifier
10. ถ้ายังไม่ผ่าน จะใช้ contextual fallback ตาม route และเนื้อคำถาม
11. `app/io_csv.py` เขียน `submission.csv.tmp` ก่อน แล้ว rename เป็น `submission.csv` แบบ atomic
12. `app/progress.py` เรียก progress หลังไฟล์ output เขียนเสร็จเท่านั้น

หมายเหตุเรื่อง log ของระบบแข่งขัน: `run.py` จะ sleep ตอนเริ่มต้น 10 วินาทีผ่านค่า `STARTUP_SLEEP_SECONDS` เพื่อให้ระบบ log ของเว็บมีเวลาจับ output ก่อนโหลดโมเดล หากต้องการปรับเวลาให้ตั้ง environment variable นี้ได้

## Contextual fallback

Fallback คือคำตอบสำรองที่ใช้เมื่อคำตอบจากโมเดลไม่ผ่าน guard หลัง rewrite หนึ่งรอบ หรือใช้เมื่อ pipeline หลักโหลดโมเดลไม่สำเร็จแล้วต้องสร้าง `submission.csv` ให้ครบ

ระบบจะไม่ใช้ประโยคกลางๆ เหมือนกันทุกคำถาม แต่จะดู route และ keyword/category จากคำถาม เช่น:

- `UNSAFE`: ปฏิเสธเฉพาะขั้นตอนอันตราย และเสนอข้อมูลปลอดภัยแทน เช่น การป้องกัน ความเสี่ยง หรือการขอความช่วยเหลือ
- `SAFE_SENSITIVE`: ตอบเป็นข้อมูลทั่วไปตามโดเมน เช่น กฎหมาย สุขภาพ การเงิน หรือ cyber defensive โดยเลี่ยงคำแนะนำเฉพาะที่เสี่ยง
- `SAFE_DIRECT`: ถ้าเป็นคำถามปลอดภัยที่รู้ pattern ชัด เช่น Linux process, coding/debug, สรุป/อธิบาย จะให้คำตอบสั้นที่ยังใช้งานได้

ตัวอย่างเช่น คำถาม `kill process ใน Linux` จะไม่ถูกปฏิเสธแบบอันตราย แต่ fallback จะอธิบายการหา PID และใช้ `kill`/`kill -9` อย่างระวัง เพราะบริบทนี้เป็นงานดูแลระบบที่ปลอดภัย

## สิ่งที่ช่วยเพิ่ม Safety

- ตรวจทั้งก่อนตอบและหลังตอบ
- แยก route ระหว่างคำถามปลอดภัย คำถามอ่อนไหว และคำถามอันตราย
- ปฏิเสธเฉพาะส่วนอันตราย ไม่ปฏิเสธทั้งคำถามโดยไม่จำเป็น
- ตัด `<think>...</think>` และ role prefix ก่อนเขียน CSV
- ใช้ fallback แบบ deterministic แต่เลือกเนื้อหาตามประเภทคำถาม เมื่อ guard ไม่ผ่านหลัง rewrite
- ถ้า ThaiSafetyClassifier error จะถือว่าไม่ปลอดภัย ไม่ปล่อยผ่านอัตโนมัติ

## สิ่งที่ช่วยรักษา Helpfulness

- คำถาม safe ใช้ prompt ที่สั่งให้ตอบตรงและครบถ้วน
- คำถาม sensitive ที่ยังตอบได้ จะตอบในระดับข้อมูลทั่วไปหรือการป้องกัน
- ถ้า output guard ไม่ผ่าน จะ rewrite หนึ่งรอบก่อน fallback เพื่อไม่เสียคำตอบดีๆ ทันที
- fallback มี template เฉพาะทางสำหรับบางหมวด เช่น Linux, coding/debug, กฎหมาย, สุขภาพ, การเงิน, cyber defensive, self-harm, weapons, drugs และ fraud
- Keyword เดี่ยวๆ ไม่ได้ทำให้ระบบปฏิเสธทันที เช่นคำว่า `kill process` ใน Linux ควรยังตอบได้

## หมายเหตุ

ผมไม่ได้รัน local test หรือ inference ในเครื่องนี้ตามที่ร้องขอไว้ เพราะงานนี้ต้องใช้ GPU และโมเดลขนาดใหญ่ ให้เอาชุดไฟล์นี้ไป build/run ในระบบที่มีทรัพยากรตรงกับโจทย์ได้เลย
