# Profile Analyser Through Smart Glasses

**Academic smart-glasses concept and tested software prototype. This repository replaces its earlier facial-expression-classification contents **at the repository owner's request**. The GitHub repository URL retains its old name; the project described here is the Smart Glasses Profile Analyser.

**Team:** S. Abhi Ram, K. Shiva Kiran, T. Abhilash; the earlier course-project version also credits P. Sai Preetham.

## Original project idea
The proposed glasses mount a camera and small processor to capture a student's face, identify the student from an enrolled database, and show their profile using a near-eye display. The reports describe five original fields: hall-ticket ID, name, phone, photograph, and hobbies.

## What actually runs
The available Python prototype:
- Detects face regions in a supplied image with OpenCV's Haar cascade.
- Looks up a **synthetic** student profile using an **explicitly supplied student ID**.
- Saves an annotated detection image locally and a JSON result.

**This is not facial identity recognition**: it cannot infer the student's ID from a face. Physical glasses, projector display, and biometric identification are not implemented.

## Install and test
Use Python 3.10+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/app.py --image /path/to/consented_photo.jpg --student-id DEMO001
```

The last command creates `outputs/detected_faces.jpg` locally and `outputs/result.json`. The sample image is **not** shipped in this repository. `data/demo_profiles.csv` is fictitious; no original student dataset is published.

## Verified outputs
Automated tests: **3 of 3 passed**. Previous supplied-image execution detected **three face regions** and generated the JSON in `outputs/result.json`. This is a face detection count, **not** a student-identification accuracy.

See [execution report](outputs/execution_report.json) and [project alignment](PROJECT_ALIGNMENT.md).

## Privacy and responsible use
The supplied XLSX files and source images contain real people's private data. They are not redistributed. This demonstration should only be used on images obtained with informed consent. Avoid using biometric identity detection or public disclosure of student profiles without a legal basis, consent, access controls, and independent evaluation.

No hardware functionality or recognition accuracy is claimed.
