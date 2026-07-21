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
  -> ถ้า classifier มองไม่เห็น response และเป็น intent ที่ตอบได้ ให้ใช้ response rules ตัดสินแทน
  -> ถ้าผ่าน ส่งคำตอบ
  -> ถ้าไม่ผ่าน rewrite ด้วย main model สูงสุดหนึ่งรอบ
  -> ThaiSafetyClassifier ตรวจซ้ำ
  -> ถ้ายังไม่ผ่าน เลือก contextual fallback ที่ผ่าน ThaiSafety
  -> ถ้ายังไม่มี candidate ผ่าน ตรวจ explicit boundary + contextual help
  -> ถ้ายังไม่ผ่านและ classifier เห็น response น้อยมาก ให้รักษา contextual answer ที่ผ่าน response rules
  -> กรณีอื่นใช้ minimal safe backstop
```

โมเดลที่ตั้งใจใช้:

| หน้าที่ | โมเดล |
|---|---|
| โมเดลตอบหลัก | `Qwen/Qwen3.6-27B-FP8` |
| Guard ภาษาไทย | `typhoon-ai/ThaiSafetyClassifier` |
| Inference engine | `vLLM` |

เหตุผลที่เลือกชุดนี้:

- Qwen3.6-27B-FP8 เป็น dense checkpoint แบบ FP8 official ของ Qwen เพื่อเพิ่ม contextual reasoning และ Helpfulness โดยใช้ `language_model_only` และ non-thinking mode สำหรับงาน text batch นี้
- ไม่โหลด Qwen3Guard เพิ่ม เพื่อคืน VRAM และเวลา inference ให้ main model กับ KV cache
- ThaiSafetyClassifier ช่วยจับความเสี่ยงจากสำนวนและบริบทภาษาไทย ซึ่ง hidden dataset มีโอกาสใช้ภาษาไทยและโจทย์หลอกเชิงวัฒนธรรม
- ThaiSafetyClassifier ตั้งค่าแบบใกล้ evaluator มากขึ้น โดยใช้รูปแบบ `input: ... output: ...`, `max_length=128` และ threshold 0.50 เพื่อให้ rewrite เกิดกับเคสเสี่ยงจริง
- ตั้ง `gpu_memory_utilization`, `max_num_seqs` และ guard batch size ให้เหลือพื้นที่สำหรับ generator กับ classifier บน GPU เดียว
- Final fallback ไม่ถูกถือว่าปลอดภัยอัตโนมัติ แต่มีหลาย candidate ต่อ intent และถูกตรวจด้วย ThaiSafety ก่อนเลือกคำตอบสุดท้าย
- ไม่มีการส่ง candidate ที่ไม่ผ่านเพียงเพราะมี harmful probability ต่ำที่สุด; probability ใช้เป็น diagnostics เท่านั้น
- Output validator ปฏิเสธอักษร Han ที่ไม่ได้มาจากคำถาม, ข้อความประเมินร่างที่หลุดจาก rewrite และปลายคำตอบแบบหัวข้อว่างหรือรายการไม่จบก่อนส่งผลลัพธ์
- Helpfulness validator ส่ง generic refusal ของ intent ที่ตอบได้กลับไป rewrite และไม่ถือว่าหัวข้ออ่อนไหวเป็นเหตุให้ปฏิเสธโดยอัตโนมัติ
- หมายเลขสายด่วนและ URL ที่ยืนยันแล้วถูกแนบให้โมเดลเฉพาะ intent ที่เกี่ยวข้อง เพื่อไม่ให้ข้อมูล 1323, 1669, 1300 หรือ GSB หลุดไปยังคำตอบคนละบริบท
- Rewrite จำกัดแค่หนึ่งรอบเพื่อควบคุมเวลา inference และลดโอกาสวนซ้ำ
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
└── scripts/
    ├── download_models.py
    ├── merge_thai_guard.py
    └── validate_submission.py
```

## Path สำคัญของระบบแข่งขัน

ห้ามเก็บโมเดลไว้ใน `/model/test` เพราะระบบแข่งขันจะ mount dataset ลับเข้ามาทับ path นี้

ตำแหน่งที่ใช้จริง:

- Input dataset: `/model/test`
- Output file: `/result/submission.csv`
- Run status file: `/result/run_status.json`
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

Dockerfile ติดตั้ง vLLM รุ่นที่รองรับ Qwen3.6 โดยตรง:

```bash
docker build -t llm-trustworthiness:qwen3.6-27b-fp8 .
```

Dockerfile pin `vLLM 0.19.0` ซึ่งเป็นรุ่นขั้นต่ำที่ Qwen แนะนำสำหรับ Qwen3.6 และ config ปิด thinking mode ผ่าน `chat_template_kwargs.enable_thinking=false` เพื่อไม่ให้ reasoning tokens ปะปนในคำตอบที่ส่งเข้า guard

สำหรับ throughput บน GPU เดียว config เปิด Multi-Token Prediction ของ Qwen3.6 ผ่าน `qwen3_next_mtp` จำนวน 2 speculative tokens, ใช้ `max_num_seqs=8` และสงวน VRAM ของ vLLM ที่ `gpu_memory_utilization=0.72` หากเครื่องปลายทางมี VRAM ไม่พอให้ปิด `speculative_config` ก่อนลด batch size

## วิธีทำงานของ pipeline

1. `run.py` โหลด config และหา input CSV ใต้ `/model/test`
2. `app/io_csv.py` อ่าน records โดยรักษา `id` และลำดับเดิม
3. `app/normalization.py` normalize ข้อความสำหรับ rule guard แต่ยังส่งคำถามต้นฉบับให้ main model
4. `app/policies/rule_guard.py` ตรวจ pattern เสี่ยงและเจตนา เช่น jailbreak, cyber abuse, verify claim, human-chatbot emotion, consent/privacy, humanitarian/cultural analysis, civic action, mental-health support, safe creative และ counter-stereotype
5. `app/risk_router.py` แบ่งเป็น `SAFE_DIRECT`, `SAFE_SENSITIVE`, หรือ `UNSAFE` พร้อมส่ง intent categories ไปยัง prompt
6. `app/inference/generator.py` ใช้ vLLM สร้างคำตอบแบบ batch
7. Output ถูกตรวจด้วย response rule check และ ThaiSafetyClassifier
8. ถ้าไม่ผ่าน จะ rewrite หนึ่งรอบด้วย prompt `prompts/rewrite_th.txt`
9. คำตอบ rewrite ถูกตรวจซ้ำด้วย ThaiSafetyClassifier
10. ถ้ายังไม่ผ่าน จะ score contextual candidates แล้วเลือกเฉพาะตัวที่ผ่าน guard หากยังไม่มีจะ score `explicit safety boundary + contextual help`
11. หาก query ยาวจน classifier เห็น response น้อยกว่า 8 tokens และ candidate ทุกตัวไม่ผ่าน ระบบจะรักษา contextual candidate ที่ผ่าน deterministic response rules; กรณีอื่นจึงใช้ minimal safe backstop
12. `app/io_csv.py` เขียน `submission.csv.tmp` ก่อน แล้ว rename เป็น `submission.csv` แบบ atomic
13. `run.py` เขียน `run_status.json` เพื่อบันทึกสถานะ pipeline, category counts, การตัดสินใจราย record, `finish_reason` ของ draft/rewrite และ fallback diagnostics
14. `app/progress.py` เรียก progress หลังไฟล์ output เขียนเสร็จเท่านั้น

หมายเหตุเรื่อง log ของระบบแข่งขัน: `run.py` จะ sleep ตอนเริ่มต้น 10 วินาทีผ่านค่า `STARTUP_SLEEP_SECONDS` เพื่อให้ระบบ log ของเว็บมีเวลาจับ output ก่อนโหลดโมเดล หากต้องการปรับเวลาให้ตั้ง environment variable นี้ได้

## Contextual fallback

Fallback คือคำตอบสำรองที่ใช้เมื่อคำตอบจากโมเดลไม่ผ่าน guard หลัง rewrite หนึ่งรอบ หรือใช้เมื่อ pipeline หลักโหลดโมเดลไม่สำเร็จแล้วต้องสร้าง `submission.csv` ให้ครบ

ระบบจะไม่ใช้ประโยคกลางๆ เหมือนกันทุกคำถาม แต่จะดู route และ intent/category จากคำถาม เช่น:

- `UNSAFE`: ปฏิเสธเฉพาะขั้นตอนอันตราย และเสนอข้อมูลปลอดภัยแทน เช่น การป้องกัน ความเสี่ยง หรือการขอความช่วยเหลือ
- `SAFE_SENSITIVE`: ตอบเป็นข้อมูลทั่วไปตามโดเมน เช่น กฎหมาย สุขภาพ การเงิน หรือ cyber defensive โดยเลี่ยงคำแนะนำเฉพาะที่เสี่ยง
- `SAFE_DIRECT`: ถ้าเป็นคำถามปลอดภัยที่รู้ pattern ชัด เช่น Linux process, coding/debug, สรุป/อธิบาย จะให้คำตอบสั้นที่ยังใช้งานได้

Intent สำคัญของ `SAFE_SENSITIVE` ได้แก่ `verify_claim`, `policy_analysis`, `cultural_analysis`, `humanitarian_analysis`, `human_chatbot_emotion`, `consent_privacy`, `responsible_reporting`, `civic_action`, `mental_health_support`, `safe_creative` และ `counter_stereotype` แต่ละ intent มี prompt guidance และ fallback candidates ของตนเอง เพื่อให้ตอบส่วนที่ปลอดภัยได้จริงแทนการปฏิเสธทั้งคำถาม

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
- คำถาม sensitive ที่เป็นการวิเคราะห์ ตรวจข่าว ขอความช่วยเหลือ งานสร้างสรรค์ หรือนโยบาย จะตอบงานนั้นโดยตรงและปฏิเสธเฉพาะรายละเอียดที่เพิ่มความสามารถในการก่อเหตุ
- ถ้า output guard ไม่ผ่าน จะ rewrite หนึ่งรอบก่อน fallback เพื่อไม่เสียคำตอบดีๆ ทันที
- คำตอบ sensitive และ rewrite ใช้ budget 224 tokens, โดยทั่วไปถูกสั่งให้จบใน 3-6 ประโยค และบันทึก `finish_reason`
- Generic refusal ใน intent ที่ตอบได้จะไม่ถูกยอมรับ และ classifier จะไม่ใช้เป็นเหตุผลเดียวในการทิ้งคำตอบเมื่อหน้าต่าง 128 tokens ไม่เห็น response เลย
- เมื่อ `finish_reason=length` ระบบจะตัดเฉพาะเศษประโยคท้ายถ้ามีประโยคสมบูรณ์ก่อนหน้า หากซ่อมไม่ได้จะไม่ยอมรับ output และส่งเข้า rewrite/fallback
- fallback มี template เฉพาะทางสำหรับข่าวลือ นโยบาย สุขภาพจิต งานสร้างสรรค์ การตอบโต้การเหมารวม รวมถึง Linux, cyber defensive, self-harm, weapons, drugs และ fraud
- Keyword เดี่ยวๆ ไม่ได้ทำให้ระบบปฏิเสธทันที เช่นคำว่า `kill process` ใน Linux ควรยังตอบได้

ข้อมูล high-stakes ที่ hard-code มีเฉพาะรายการที่ตรวจจากหน่วยงานทางการ: 1323 เป็นสายด่วนสุขภาพจิต 24 ชั่วโมง, 1669 เป็นระบบการแพทย์ฉุกเฉิน, 1300 เป็นศูนย์ช่วยเหลือสังคม และเว็บไซต์ธนาคารออมสินคือ `https://www.gsb.or.th/` ระบบถูกสั่งไม่ให้สร้าง URL หมายเลขโทรศัพท์ หรือข้อมูลการแพทย์อื่นจากการคาดเดา

## หมายเหตุ

รัน compile checks ในเครื่องนี้แล้ว แต่ยังไม่ได้รัน inference เต็ม 1,889 ข้อ เพราะต้องใช้ GPU และ model weights ตาม production
