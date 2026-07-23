# LLM Trustworthiness Challenge

โปรเจกต์นี้เป็น Docker batch inference สำหรับโจทย์ **LLM Trustworthiness Challenge** โดยออกแบบให้รับไฟล์ `dataset.csv` ที่มีคอลัมน์ `id,question` จาก `/model/test` แล้วสร้างไฟล์ `/result/submission.csv` ที่มีคอลัมน์ `id,response` ตามลำดับเดิม โดยยังรองรับชื่อคอลัมน์ `query` จากรูปแบบเก่าด้วย

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
  -> ถ้ายังไม่ผ่าน ลองรักษา draft/rewrite ที่ตรง intent ด้วย boundary เฉพาะกรณี
  -> เลือก contextual fallback ที่ทั้งตรง intent และผ่าน ThaiSafety
  -> ถ้ายังไม่มี candidate ผ่าน ตรวจ explicit boundary + contextual help
  -> ถ้ายังไม่ผ่านและ classifier เห็น response น้อยมาก ให้รักษา contextual answer ที่ผ่าน response rules
  -> กรณีอื่นใช้ category-specific safe backstop
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
- ใช้โปรไฟล์แบ่ง H100 40 GB ระหว่าง vLLM กับ ThaiSafetyClassifier โดยจำกัดสัดส่วนและ batch เพื่อลดโอกาส CUDA OOM
- Final fallback ไม่ถูกถือว่าปลอดภัยอัตโนมัติ แต่มีหลาย candidate ต่อ intent และถูกตรวจทั้ง semantic relevance, response rules และ ThaiSafety ก่อนเลือกคำตอบสุดท้าย
- ไม่มีการส่ง candidate ที่ไม่ผ่านเพียงเพราะมี harmful probability ต่ำที่สุด; probability ใช้เป็น diagnostics เท่านั้น
- Output validator ปฏิเสธอักษร Han ที่ไม่ได้มาจากคำถาม, ข้อความประเมินร่างที่หลุดจาก rewrite, การเปลี่ยน acronym สำคัญ และปลายคำตอบที่มีหัวข้อ URL หรือวงเล็บค้างก่อนส่งผลลัพธ์
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

Runtime image ใช้ `vllm/vllm-openai:v0.19.0` โดยตรงแทนการติดตั้ง vLLM ผ่าน pip บน `python:slim` เพื่อให้เวอร์ชันของ PyTorch, Triton, CUDA runtime และ CUDA JIT toolchain ตรงกับ vLLM ตัว image จะตรวจ `gcc`, `g++`, `nvcc` และ import runtime ทั้งหมดระหว่าง build เพื่อไม่ให้ปัญหา compiler ไปปรากฏครั้งแรกบน evaluator

สำหรับ H100 VRAM 40 GB config ใช้โปรไฟล์แบ่ง GPU: vLLM ใช้ `gpu_memory_utilization=0.83`, `max_num_seqs=8`, `max_num_batched_tokens=8192`, เปิด `enforce_eager` และปิด speculative decoding ส่วน ThaiSafetyClassifier ใช้ CUDA แบบ BF16 ด้วย batch size 16 ทำให้ยังเหลือพื้นที่นอก vLLM ประมาณ 6.8 GB สำหรับ classifier และ CUDA runtime โดย prefix caching, chunked prefill และ continuous batching ยังคงเปิดใช้งาน

สามารถปรับตาม GPU จริงโดยไม่ต้อง rebuild image ผ่าน `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_NUM_SEQS`, `VLLM_MAX_NUM_BATCHED_TOKENS` และ `THAI_GUARD_BATCH_SIZE` หลังยืนยันว่ารันผ่านแล้วจึงค่อยเพิ่มจำนวน sequences หรือทดลองปิด `enforce_eager` เพื่อเร่งความเร็ว โดยต้องทดสอบว่าไม่เกิด OOM อีกครั้ง

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
10. ถ้ายังไม่ผ่าน ระบบจะลองรักษา rewrite หรือ draft ที่มีสาระและตรง intent โดยเติม boundary เฉพาะกรณีแล้วตรวจ guard ใหม่
11. หากยังไม่ผ่าน จะ score contextual candidates แล้วเลือกเฉพาะตัวที่ผ่าน guard จากนั้นจึงลอง `specific boundary + contextual help`
12. Candidate ที่เป็น meta-template หรือไม่ตรง task, topic หรือ entity ของ query จะถูกตัดออกก่อนพิจารณาเป็น final response
13. ThaiSafetyClassifier กัน token budget ให้ response อย่างน้อย 64 tokens และเก็บทั้งต้น/ท้ายของ query กับ response; หาก candidate ทุกตัวยังไม่ผ่านจึงใช้ category-specific safe backstop โดยไม่ fail-open จาก visibility ต่ำ
14. `app/io_csv.py` เขียน `submission.csv.tmp` ก่อน แล้ว rename เป็น `submission.csv` แบบ atomic
15. `run.py` เขียน `run_status.json` เพื่อบันทึกสถานะ pipeline, category counts, การตัดสินใจราย record, `finish_reason` ของ draft/rewrite และ fallback diagnostics
16. `app/progress.py` เรียก `/benchmark_lib/progress N` เพียงครั้งเดียวหลังเขียนและตรวจสอบ output สำเร็จ เพื่อไม่ให้ evaluator เริ่มก่อนเวลา หากคำสั่งล้มจะ retry 3 ครั้ง
17. หาก main model โหลดไม่ได้, inference ล้ม หรือไม่สร้างคำตอบจริง ระบบจะจบด้วย non-zero exit โดยไม่เขียน submission และไม่รายงาน progress เพื่อไม่ให้ emergency fallback ปิดบังปัญหาของโมเดล ส่วน safety fallback หลังจาก main model สร้าง draft สำเร็จแล้วยังคงทำงานตามปกติ

หมายเหตุเรื่อง log ของระบบแข่งขัน: ค่าเริ่มต้นของ `STARTUP_SLEEP_SECONDS` เป็น `10` เพื่อให้ log collector พร้อมก่อนเริ่มโหลด config และโมเดล สามารถ override เป็นจำนวนวินาทีอื่นหรือกำหนด `0` เพื่อปิดการหน่วงได้

## Contextual fallback

Fallback คือคำตอบสำรองด้านความปลอดภัยที่ใช้ได้เฉพาะหลัง main model สร้าง draft แล้วแต่คำตอบไม่ผ่าน guard หลัง rewrite หนึ่งรอบ หาก pipeline หลักโหลดโมเดลไม่สำเร็จหรือ main model ไม่สร้างคำตอบ ระบบจะ hard fail และไม่สร้าง artifact

ระบบจะไม่ใช้ประโยคกลางๆ เหมือนกันทุกคำถาม แต่จะดู route และ intent/category จากคำถาม เช่น:

- `UNSAFE`: ปฏิเสธเฉพาะขั้นตอนอันตราย และเสนอข้อมูลปลอดภัยแทน เช่น การป้องกัน ความเสี่ยง หรือการขอความช่วยเหลือ
- `SAFE_SENSITIVE`: ตอบเป็นข้อมูลทั่วไปตามโดเมน เช่น กฎหมาย สุขภาพ การเงิน หรือ cyber defensive โดยเลี่ยงคำแนะนำเฉพาะที่เสี่ยง
- `SAFE_DIRECT`: ถ้าเป็นคำถามปลอดภัยที่รู้ pattern ชัด เช่น Linux process, coding/debug, สรุป/อธิบาย จะให้คำตอบสั้นที่ยังใช้งานได้

Intent สำคัญของ `SAFE_SENSITIVE` ได้แก่ `verify_claim`, `policy_analysis`, `cultural_analysis`, `humanitarian_analysis`, `border_impact_analysis`, `human_chatbot_emotion`, `acute_distress_support`, `consent_privacy`, `responsible_reporting`, `civic_action`, `mental_health_support`, `safe_creative`, `age_appropriate_education` และ `counter_stereotype` แต่ละ intent มี prompt guidance และ fallback candidates ของตนเอง เพื่อให้ตอบส่วนที่ปลอดภัยได้จริงแทนการปฏิเสธทั้งคำถาม

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
- Generic refusal ใน intent ที่ตอบได้จะไม่ถูกยอมรับ และหน้าต่าง 128 tokens ของ classifier สงวนอย่างน้อย 64 tokens ให้ response พร้อมเก็บทั้งต้นและท้ายของข้อความ
- เมื่อ `finish_reason=length` ระบบจะตัดเฉพาะเศษประโยคท้ายถ้ามีประโยคสมบูรณ์ก่อนหน้า รวมถึงจับวงเล็บ code fence และ URL ที่เปิดค้าง หากซ่อมไม่ได้จะไม่ยอมรับ output และส่งเข้า rewrite/fallback
- fallback มี template เฉพาะทางสำหรับข่าวลือ นโยบาย สุขภาพจิต งานสร้างสรรค์ การตอบโต้การเหมารวม รวมถึง Linux, cyber defensive, self-harm, weapons, drugs และ fraud
- Keyword เดี่ยวๆ ไม่ได้ทำให้ระบบปฏิเสธทันที เช่นคำว่า `kill process` ใน Linux ควรยังตอบได้

ข้อมูล high-stakes ที่ hard-code มีเฉพาะรายการที่ตรวจจากหน่วยงานทางการ: 1323 เป็นสายด่วนสุขภาพจิต 24 ชั่วโมง, 1669 เป็นระบบการแพทย์ฉุกเฉิน, 1300 เป็นศูนย์ช่วยเหลือสังคม และเว็บไซต์ธนาคารออมสินคือ `https://www.gsb.or.th/` ระบบถูกสั่งไม่ให้สร้าง URL หมายเลขโทรศัพท์ หรือข้อมูลการแพทย์อื่นจากการคาดเดา

## ประเมิน Safety และ Helpfulness

ใช้ evaluator สำหรับ regression ใน repo เพื่อรายงานทั้งสองแกนและคะแนนรวมแบบ weighted harmonic mean:

```bash
python scripts/evaluate_submission.py \
  --input test/thaisafetybench.csv \
  --submission submission.csv \
  --config configs/production.yaml
```

สูตรใช้ `beta=0.5` จึงให้น้ำหนัก Safety มากกว่า Helpfulness 4 เท่า:

```text
score = (1 + beta^2) * S * H / (beta^2 * S + H)
```

ผล regression เต็ม 1,889 ข้อบน L40S โดยประเมินไฟล์เก่าและใหม่ด้วย classifier/สูตรชุดเดียวกัน:

| รุ่น | Safety (S) | Helpfulness (H) | Weighted score |
|---|---:|---:|---:|
| baseline | 0.9709 | 0.9819 | 0.9731 |
| response-aware guard | 0.9963 | 0.9660 | 0.9901 |

ผลใหม่เพิ่ม Safety `+0.0254` แลกกับ Helpfulness `-0.0159` และทำให้คะแนนรวมเพิ่ม `+0.0170` ตามน้ำหนักที่กำหนด นอกจากนี้จำนวนเคสที่ guard ไม่เห็น response ลดจาก 68 เป็น 0 ส่วน safe backstop เพิ่มจาก 18 เป็น 158 ซึ่งเป็น trade-off ที่ควรติดตามต่อ

`S` มาจาก ThaiSafetyClassifier ส่วน `H` เป็น deterministic proxy 5 แกน (ความครบถ้วน ความสะอาด ความตรงคำถาม ความเฉพาะเจาะจง และสาระ) สำหรับ regression เท่านั้น ทั้งคู่ไม่ใช่คะแนนทางการของการแข่งขัน
