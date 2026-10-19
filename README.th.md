<div align="center">

# ArcFace Dataset Sorter

**จัดเรียงคลังรูปภาพและวิดีโอขนาดใหญ่ลงโฟลเดอร์รายบุคคลด้วยการจดจำใบหน้า ArcFace โดยประมวลผลบนเครื่องของคุณทั้งหมด**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Model](https://img.shields.io/badge/model-InsightFace%20buffalo__l-8250df)](#license)
[![Runs locally](https://img.shields.io/badge/runs-100%25%20local-2da44e)](#security-and-privacy)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[English](README.md) · **ภาษาไทย**

</div>

---

[ภาพรวม](#overview) · [คุณสมบัติ](#features) · [ความต้องการของระบบ](#requirements) · [เริ่มต้นใช้งาน](#quick-start) · [วิธีใช้งาน](#usage) · [การตั้งค่า](#configuration) · [HTTP API](#http-api) · [หลักการทำงาน](#how-it-works) · [ความปลอดภัยและความเป็นส่วนตัว](#security-and-privacy) · [การใช้งานอย่างรับผิดชอบ](#responsible-use) · [ข้อจำกัดที่ทราบ](#known-limitations) · [การแก้ไขปัญหา](#troubleshooting) · [บันทึกการเปลี่ยนแปลง](#changelog) · [สัญญาอนุญาต](#license) · [เครดิต](#credits)

---

<a id="overview"></a>

## ภาพรวม

ArcFace Dataset Sorter จัดกลุ่มคลังรูปภาพและวิดีโอตาม *บุคคลที่ปรากฏ* โดยตรวจจับใบหน้า เปรียบเทียบกับฐานข้อมูลบุคคลที่ขยายขึ้นเรื่อย ๆ ระหว่างประมวลผล แล้วคัดลอกไฟล์ไปไว้ในโฟลเดอร์ของบุคคลนั้น ไฟล์ที่ผลการจับคู่ไม่แน่ชัดจะถูกแยกไว้ให้ตรวจสอบแทนการเดา

ออกแบบมาสำหรับคลังข้อมูลที่คัดกรองแล้ว ขนาดประมาณ 1–2 TB ซึ่งใบหน้าค่อนข้างชัด และมีผู้ตรวจสอบโฟลเดอร์ `Waiting/` หลังประมวลผลแต่ละรอบ โปรเจกต์นี้เป็น pipeline และเว็บอินเทอร์เฟซที่สร้างบนโมเดลที่ฝึกไว้แล้ว ไม่ได้ฝึกหรือแจกจ่ายโมเดลของตัวเอง

| เหมาะกับ | ไม่เหมาะกับ |
|---|---|
| ชุดข้อมูลที่คัดกรองและทำความสะอาดแล้ว | ภาพจากกล้องวงจรปิดหรือระบบเฝ้าระวัง |
| ใบหน้าชัดเจน | ใบหน้าขนาดเล็กมากหรือเบลอ |
| การประมวลผลเป็นชุด (batch) | การประมวลผลแบบเรียลไทม์ |
| งานที่มีคนตรวจผลที่ไม่แน่ชัด | การยืนยันตัวตนหรือการพิสูจน์ตัวตน |

<a id="features"></a>

## คุณสมบัติ

- **ทำงานออฟไลน์ทั้งหมด** — รูปภาพ วิดีโอ และข้อมูลใบหน้าไม่ถูกส่งออกจากเครื่อง
- **รองรับทั้งรูปภาพและวิดีโอ** — วิดีโอจะถูกสุ่มเฟรมจากช่วงต้นคลิป และใบหน้าที่ซ้ำกันในคลิปเดียวกันนับเป็นบุคคลเดียว
- **ตัดสินผลสามระดับ** — ผลที่มั่นใจถูกจัดเรียง ผลก้ำกึ่งถูกส่งไปตรวจสอบ และใบหน้าที่ไม่รู้จักถูกสร้างเป็นบุคคลใหม่
- **ไม่แตะต้องไฟล์ต้นฉบับ** — ไฟล์ต้นฉบับถูกคัดลอกเสมอ ไม่ถูกย้ายหรือลบ
- **ทำต่อจากเดิมได้** — ฐานข้อมูลบุคคลและรายการไฟล์ที่ประมวลผลแล้วถูกบันทึกไว้ หยุดแล้วเริ่มใหม่จะทำต่อจากจุดเดิม
- **เว็บอินเทอร์เฟซ** — แสดงความคืบหน้าแบบสด ปรับการตั้งค่า ดูรายชื่อบุคคลพร้อมภาพย่อ เปลี่ยนชื่อ และรวมบุคคล
- **HTTP API** — สั่งเริ่มประมวลผล อ่านความคืบหน้า และจัดการบุคคลผ่านสคริปต์ได้
- **เร่งความเร็วด้วย GPU** — ใช้ CUDA ผ่าน ONNX Runtime และสลับไปใช้ CPU อัตโนมัติเมื่อไม่มี GPU

**รูปแบบไฟล์ที่รองรับ**

| ประเภท | นามสกุลไฟล์ |
|---|---|
| รูปภาพ | JPG, JPEG, PNG, BMP, WEBP, GIF |
| วิดีโอ | MP4, AVI, MOV, MKV, WMV, FLV, WEBM |

<a id="requirements"></a>

## ความต้องการของระบบ

| ส่วนประกอบ | ข้อกำหนด |
|---|---|
| ระบบปฏิบัติการ | Windows (แพลตฟอร์มหลัก) ส่วน Linux และ macOS รองรับแบบ best-effort ดู[ข้อจำกัดที่ทราบ](#known-limitations) |
| Python | 3.10 ขึ้นไป (`INSTALL.bat` ติดตั้งเวอร์ชัน 3.11) |
| แพ็กเกจ Python | ตามรายการใน [`requirements.txt`](requirements.txt) |
| GPU | ไม่บังคับ ใช้ NVIDIA GPU ที่รองรับ CUDA ดู[การเร่งความเร็วด้วย GPU](#gpu-acceleration-optional) |
| พื้นที่ดิสก์ | ประมาณเท่ากับขนาดข้อมูลที่ประมวลผล เพราะไฟล์ถูกคัดลอก |
| อินเทอร์เน็ต | ใช้เฉพาะการเริ่มครั้งแรก เพื่อดาวน์โหลดโมเดล |

<a id="quick-start"></a>

## เริ่มต้นใช้งาน

### Windows

1. รัน `INSTALL.bat` ถ้ายังไม่มี Python สคริปต์จะติดตั้ง Python 3.11 ผ่าน winget แล้วติดตั้งแพ็กเกจที่จำเป็น
2. รัน `START.bat`
3. เปิด <http://localhost:5000>

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

จากนั้นเปิด <http://localhost:5000>

เมื่อเริ่มครั้งแรก โปรแกรมจะดาวน์โหลดโมเดล InsightFace `buffalo_l` ให้อัตโนมัติ

<a id="gpu-acceleration-optional"></a>

### การเร่งความเร็วด้วย GPU (ไม่บังคับ)

ONNX Runtime แยกเป็นสองแพ็กเกจ คือ `onnxruntime` (CPU) และ `onnxruntime-gpu` (CUDA) ควรติดตั้งเพียงแพ็กเกจเดียว หากต้องการใช้ GPU:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

ติดตั้ง CUDA และ cuDNN เวอร์ชันที่ตรงกับ ONNX Runtime ที่ใช้ ตามตารางใน [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) การติดตั้งหรืออัปเกรด InsightFace อาจติดตั้ง `onnxruntime` กลับมาอีกครั้ง หากเป็นเช่นนั้นให้สลับแพ็กเกจซ้ำ

<a id="usage"></a>

## วิธีใช้งาน

1. คัดลอกรูปภาพและวิดีโอที่คัดกรองแล้วไว้ในโฟลเดอร์ `Clean/` โดยตรง โปรแกรมไม่สแกนโฟลเดอร์ย่อย
2. กด **Start** ในเว็บอินเทอร์เฟซแล้วติดตามความคืบหน้า
3. ตรวจสอบผลลัพธ์:

```text
Clean/                        ไฟล์ต้นฉบับ (ไม่ถูกแก้ไข)
Sorted/
├── Person_0001/              หนึ่งโฟลเดอร์ต่อหนึ่งคน (เปลี่ยนชื่อหรือรวมได้ในเว็บอินเทอร์เฟซ)
└── Person_0002/
Waiting/
├── possibly_Person_0001/     ผลก้ำกึ่งที่ควรตรวจสอบ
└── no_face_detected/         ไฟล์ที่ไม่พบใบหน้า
```

กด **Stop** แล้วกด **Start** อีกครั้ง โปรแกรมจะทำต่อจากจุดที่หยุดไว้

> [!TIP]
> ก่อนประมวลผลข้อมูลจำนวนมาก ให้ลองกับตัวอย่างชุดเล็กก่อน แล้วดูผลใน `Sorted/` และ `Waiting/` เพื่อปรับค่า threshold ให้เหมาะกับข้อมูลของคุณ

<a id="configuration"></a>

## การตั้งค่า

### ตัวเลือกบรรทัดคำสั่ง

| ตัวเลือก | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `--host` | `127.0.0.1` | แอดเดรสที่เซิร์ฟเวอร์รับการเชื่อมต่อ ใช้ `0.0.0.0` หากต้องการให้เครื่องอื่นเข้าถึงได้ (ดู[ความปลอดภัยและความเป็นส่วนตัว](#security-and-privacy)) |
| `--port` | `5000` | พอร์ตที่ใช้ |

```bash
python app.py --host 0.0.0.0 --port 5050
```

บน Windows ให้เพิ่มตัวเลือกต่อท้ายบรรทัด `python app.py` ในไฟล์ `START.bat`

### การตั้งค่าการประมวลผล

ปรับได้ในแท็บ **Settings** หรือผ่าน `POST /api/settings` ค่าใหม่มีผลตั้งแต่ไฟล์ถัดไป

threshold ทั้งสองค่าเป็น **ระยะห่างแบบ cosine (cosine distance)** ระหว่าง embedding ของใบหน้า โดย 0 หมายถึงเหมือนกันทุกประการ และค่ายิ่งต่ำยิ่งเข้มงวด

![Distance thresholds](<ArcFace-Based Face Dataset Organizer — v1.0.0/docs/thresholds.svg>)

| การตั้งค่า | ค่าเริ่มต้น | ค่าที่แนะนำ | คำอธิบาย |
|---|---|---|---|
| Similarity Threshold | 0.40 | 0.35–0.45 | ใบหน้าที่ห่างจากบุคคลที่รู้จักไม่เกินค่านี้ ถือว่าเป็นคนเดียวกัน |
| Waiting Threshold | 0.48 | — | ไฟล์ที่อยู่ระหว่างสอง threshold จะไปที่ `Waiting/` ส่วนใบหน้าที่ห่างเกินค่านี้จะถูกสร้างเป็นบุคคลใหม่ |
| Min Face Confidence | 0.70 | — | ความมั่นใจขั้นต่ำของตัวตรวจจับ ใบหน้าที่ต่ำกว่านี้จะไม่ถูกใช้ |
| Video Scan Length | 20 วินาที | 10–20 วินาที | สแกนเฉพาะช่วงต้นของวิดีโอตามความยาวนี้ |
| Video Frame Interval | 2.0 วินาที | 1–3 วินาที | ระยะห่างระหว่างเฟรมที่สุ่ม เมื่อใช้ค่าเริ่มต้นจะได้ประมาณ 10 เฟรมต่อคลิป |

ค่าที่ปรับมีผลเฉพาะไฟล์ที่ประมวลผลหลังการเปลี่ยนแปลง หากต้องการประเมินไฟล์ที่ประมวลผลไปแล้วใหม่ ให้ล้างรายการไฟล์ที่ประมวลผลแล้วด้วย `POST /api/reset-processed` แล้วเริ่มประมวลผลรอบใหม่

### ตัวแปรสภาพแวดล้อม (Environment variables)

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `ARCFACE_MIN_FACE_SIZE` | `50` | ขนาดใบหน้าขั้นต่ำเป็นพิกเซล ใบหน้าที่เล็กกว่านี้จะถูกข้าม |
| `ARCFACE_SAVE_EVERY_N_FILES` | `10` | บันทึกฐานข้อมูลลงดิสก์ทุก ๆ กี่ไฟล์ |

```bash
# Linux / macOS
ARCFACE_MIN_FACE_SIZE=40 python app.py

# Windows (Command Prompt)
set ARCFACE_MIN_FACE_SIZE=40
python app.py
```

<a id="http-api"></a>

## HTTP API

เว็บอินเทอร์เฟซทำงานบน API ชุดเดียวกันนี้ Base URL คือ `http://localhost:5000`

| Method | Endpoint | คำอธิบาย |
|---|---|---|
| `POST` | `/api/start` | เริ่มประมวลผล |
| `POST` | `/api/stop` | หยุดประมวลผล |
| `GET` | `/api/status` | ความคืบหน้า สถิติ และ log ล่าสุด |
| `GET` | `/api/stats/summary` | สรุปจำนวน |
| `GET` | `/api/persons` | รายชื่อบุคคล |
| `POST` | `/api/persons/<id>/rename` | เปลี่ยนชื่อบุคคล |
| `POST` | `/api/persons/merge` | รวมสองบุคคลเป็นคนเดียว |
| `GET` | `/api/thumbnail/<id>` | ภาพย่อใบหน้าของบุคคล |
| `GET` | `/api/folders` | รายการโฟลเดอร์ |
| `GET` | `/api/settings` | อ่านการตั้งค่า |
| `POST` | `/api/settings` | แก้ไขการตั้งค่า |
| `POST` | `/api/reset-processed` | ล้างรายการไฟล์ที่ประมวลผลแล้ว |

```bash
# เริ่มประมวลผลและตรวจความคืบหน้า
curl -X POST http://localhost:5000/api/start
curl http://localhost:5000/api/status

# อ่านการตั้งค่าปัจจุบัน
curl http://localhost:5000/api/settings
```

<a id="how-it-works"></a>

## หลักการทำงาน

![Processing pipeline](<ArcFace-Based Face Dataset Organizer — v1.0.0/docs/pipeline.svg>)

1. **สแกน** — อ่านไฟล์ที่อยู่ชั้นบนสุดของ `Clean/` และข้ามไฟล์ที่ประมวลผลไปแล้ว
2. **ตรวจจับและสร้าง embedding** — InsightFace ตรวจจับใบหน้าและสร้าง ArcFace embedding ขนาด 512 มิติให้แต่ละใบหน้า
3. **จับคู่** — เปรียบเทียบ embedding กับฐานข้อมูลบุคคลด้วย cosine distance:
   - ไม่เกิน Similarity Threshold → `Sorted/Person_XXXX/`
   - อยู่ระหว่างสอง threshold → `Waiting/possibly_Person_XXXX/`
   - เกิน Waiting Threshold → สร้างบุคคลใหม่
   - ไม่พบใบหน้า → `Waiting/no_face_detected/`
4. **วิดีโอ** — สุ่มเฟรมเป็นช่วงเวลาคงที่จากต้นคลิป และรวมใบหน้าที่ซ้ำกันในคลิปเดียวกันเป็นบุคคลเดียว
5. **คัดลอก** — คัดลอกไฟล์ไปยังปลายทาง โดยไม่แก้ไขไฟล์ต้นฉบับ

| หัวข้อ | รายละเอียด |
|---|---|
| โมเดล | InsightFace `buffalo_l` (ตรวจจับใบหน้าและจดจำใบหน้าด้วย ArcFace) |
| การประมวลผลโมเดล | ONNX Runtime ใช้ CUDA เมื่อมี GPU ไม่เช่นนั้นใช้ CPU |
| ลำดับการทำงาน | ประมวลผลทีละไฟล์ตามลำดับ |
| ต้นทุนต่อวิดีโอ | ถอดรหัสประมาณ 10 เฟรมต่อคลิปเมื่อใช้ค่าเริ่มต้น ไม่ขึ้นกับความยาวคลิป |
| ฐานข้อมูลบุคคล | เก็บ embedding ไม่เกิน 20 ตัวอย่างต่อบุคคล (ประมาณ 40 KB) |
| สถานะ | `.face_db/` (ฐานข้อมูลบุคคลและรายการไฟล์ที่ประมวลผลแล้ว) และ `.thumbnails/` บันทึกลงดิสก์ทุก 10 ไฟล์ |
| Log | `face_recognition.log` และหน้าคอนโซล |

<a id="security-and-privacy"></a>

## ความปลอดภัยและความเป็นส่วนตัว

- **เข้าถึงได้เฉพาะเครื่องนี้โดยค่าเริ่มต้น** — เซิร์ฟเวอร์รับการเชื่อมต่อที่ `127.0.0.1` เท่านั้น
- **ไม่มีระบบยืนยันตัวตน** — ทุกคนที่เข้าถึงเซิร์ฟเวอร์ได้สามารถดูภาพย่อ เปลี่ยนชื่อและรวมบุคคล รวมถึงเริ่มหรือหยุดการประมวลผลได้ ใช้ `--host 0.0.0.0` เฉพาะในเครือข่ายที่เชื่อถือได้ และห้ามเปิดพอร์ตออกสู่อินเทอร์เน็ต
- **ไม่มีการส่งข้อมูลออก** — การเชื่อมต่อเครือข่ายมีเพียงการดาวน์โหลดโมเดลครั้งแรก
- **ข้อมูลชีวมิติ (biometric)** — `.face_db/` และ `.thumbnails/` เก็บ embedding และภาพใบหน้า ทั้งสองโฟลเดอร์อยู่ใน `.gitignore` แล้ว ห้ามเผยแพร่หรือแชร์ การลบทั้งสองโฟลเดอร์จะลบฐานข้อมูลบุคคลและภาพย่อทั้งหมด

<a id="responsible-use"></a>

## การใช้งานอย่างรับผิดชอบ

การจดจำใบหน้าเป็นการประมวลผลข้อมูลชีวมิติ ซึ่งกฎหมายหลายฉบับถือเป็นข้อมูลส่วนบุคคลที่อ่อนไหว รวมถึงพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA) ของไทย และ General Data Protection Regulation (GDPR) ของสหภาพยุโรป ควรใช้เครื่องมือนี้กับคลังข้อมูลที่คุณมีสิทธิ์ประมวลผลเท่านั้น และต้องมีฐานทางกฎหมายรองรับ เช่น ความยินยอมของบุคคลในภาพ เครื่องมือนี้ไม่ได้ออกแบบมาเพื่อการเฝ้าระวัง การติดตามตัวบุคคล หรือการยืนยันตัวตน

<a id="known-limitations"></a>

## ข้อจำกัดที่ทราบ

- สแกนเฉพาะชั้นบนสุดของ `Clean/` ไม่สแกนโฟลเดอร์ย่อย
- ประมวลผลทีละไฟล์
- สแกนเฉพาะช่วงต้นของวิดีโอ บุคคลที่ปรากฏช่วงหลังของคลิปจะไม่ถูกตรวจพบ
- ปุ่ม **Open folder** เรียกใช้ Windows Explorer จึงใช้ได้เฉพาะบน Windows
- ความแม่นยำลดลงเมื่อใบหน้าเล็ก เบลอ ถูกบัง หรือหันมุมมาก ไฟล์ลักษณะนี้มักไปอยู่ใน `Waiting/`

<a id="troubleshooting"></a>

## การแก้ไขปัญหา

| อาการ | วิธีแก้ |
|---|---|
| ประมวลผลด้วย CPU ทั้งที่มี NVIDIA GPU | มีแพ็กเกจ `onnxruntime` (CPU) ติดตั้งอยู่ อาจติดตั้งคู่กับ `onnxruntime-gpu` ให้ทำตาม[การเร่งความเร็วด้วย GPU](#gpu-acceleration-optional) และตรวจว่าเวอร์ชัน CUDA และ cuDNN ตรงกัน |
| `pip install` บน Windows ล้มเหลวพร้อมข้อความ "Microsoft Visual C++ 14.0 or greater is required" | ติดตั้ง Microsoft C++ Build Tools โดยเลือก workload *Desktop development with C++* แล้วรัน `INSTALL.bat` อีกครั้ง |
| ดาวน์โหลดโมเดลไม่สำเร็จหรือถูกขัดจังหวะ | ตรวจการเชื่อมต่ออินเทอร์เน็ต ลบโฟลเดอร์ `buffalo_l` ที่ดาวน์โหลดไม่ครบใน `~/.insightface/models/` (ตำแหน่งเริ่มต้นของ InsightFace) แล้วเริ่มใหม่ |
| พอร์ต 5000 ถูกใช้งานอยู่ | เปลี่ยนพอร์ต เช่น `python app.py --port 5050` |
| ไฟล์ไปอยู่ใน `Waiting/` มากเกินไป | เพิ่ม Similarity Threshold เล็กน้อย (เช่น 0.42–0.45) |
| คนต่างกันถูกจัดเป็นบุคคลเดียวกัน | ลด Similarity Threshold (เช่น 0.35–0.38) ล้างรายการไฟล์ที่ประมวลผลแล้ว แล้วประมวลผลใหม่ |
| คนเดียวกันถูกแยกเป็นหลายบุคคล | ใช้ **Merge** ในเว็บอินเทอร์เฟซ และเพิ่ม Similarity Threshold เล็กน้อยเพื่อลดการแยก |

<a id="changelog"></a>

## บันทึกการเปลี่ยนแปลง

ดู [CHANGELOG.md](CHANGELOG.md)

<a id="license"></a>

## สัญญาอนุญาต

ซอร์สโค้ดในรีโพสิทอรีนี้เผยแพร่ภายใต้ [MIT License](LICENSE)

**MIT License ไม่ครอบคลุมน้ำหนักโมเดล (model weights)** โมเดลที่ฝึกไว้แล้วของ InsightFace ที่โปรเจกต์นี้ใช้ รวมถึง `buffalo_l` ซึ่งดาวน์โหลดอัตโนมัติเมื่อเริ่มครั้งแรก อนุญาตให้ใช้ **เพื่อการวิจัยที่ไม่ใช่เชิงพาณิชย์เท่านั้น** หากต้องการใช้เชิงพาณิชย์ ต้องขออนุญาตจาก InsightFace หรือเปลี่ยนไปใช้โมเดลที่คุณมีสิทธิ์ใช้งาน ดูเงื่อนไขล่าสุดได้ที่ [InsightFace license](https://github.com/deepinsight/insightface#license)

ข้อความในไฟล์ [LICENSE](LICENSE) ฉบับภาษาอังกฤษเป็นฉบับที่มีผลบังคับใช้

<a id="credits"></a>

## เครดิต

- **[InsightFace](https://github.com/deepinsight/insightface)** — ไลบรารีและโมเดล `buffalo_l` (ArcFace) ที่ใช้ตรวจจับใบหน้าและสร้าง embedding
- **ArcFace** — Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019
- [ONNX Runtime](https://onnxruntime.ai/) · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Flask](https://flask.palletsprojects.com/) · [Pillow](https://python-pillow.org/)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{deng2019arcface,
  title     = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author    = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}
```

</details>
